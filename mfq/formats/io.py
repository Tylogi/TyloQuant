"""MFQ file serialization and deserialization.

Two API levels:

1. **Codec level** -- :func:`pack_nint`/:func:`unpack_nint` pack one
   :class:`~mfq.quantize.nint_quant.NintTensor` into a self-describing byte stream.
2. **File level** -- :func:`save`/:func:`load` write multiple named tensors to one ``.mfq`` file.

Binary layout
----------
::

    [FileHeader]
        magic[4] = "MFQ1"
        version  : uint32
        arch     : len(uint32) + utf8
        num_tensors : uint32
    [TensorRecord] × num_tensors
        name        : len(uint32) + utf8
        dtype       : len(uint32) + utf8   # "NINT4" / "NINT5" ...
        blob_nbytes : uint64
    [blob 0][blob 1]...                     # Compact tensor data in record order

Inside a little-endian NINT blob::

    bits(u8) sub_bits(u8) groupsize(i32) axis(i32) neuron_len(i32)
    ndim(u32) shape(ndim×i64) out(u32) ng(u32)
    neuron_scale(out×f16) neuron_min(out×f16)
    sub_scale(out·ng×sub_bits packed) sub_min(out·ng×sub_bits packed)
    q(out·ng·gs×bits packed)

Version 2 and later use bitstream storage; the loader retains read compatibility with legacy uint-storage blobs.
"""

from __future__ import annotations

import json
import mmap
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np

from mfq.formats.assets import ASSET_DTYPE
from mfq.formats.header import MFQ_MAGIC, FileHeader
from mfq.formats.moe import NintMoePool, NintMoeTensor
from mfq.formats.mx import MX_DTYPES, MxTensor, pack_mx, unpack_mx
from mfq.formats.nepq import NepqTensor, pack_nepq, rotation_signs, unpack_nepq
from mfq.formats.nint import NintSpec, _uint_dtype
from mfq.formats.nint8_zero import (
    Nint8ZeroTensor,
    pack_nint8_zero,
    unpack_nint8_zero,
)
from mfq.formats.npq0_l import Npq0LTensor, pack_npq0_l, unpack_npq0_l
from mfq.formats.npq0_s import Npq0STensor, pack_npq0_s, unpack_npq0_s
from mfq.formats.nvq import NvqJscTensor, NvqTensor, pack_nvq, unpack_nvq
from mfq.formats.nvq1_l import Nvq1LTensor, pack_nvq1_l, unpack_nvq1_l
from mfq.formats.nvq1_s import Nvq1STensor, pack_nvq1_s, unpack_nvq1_s
from mfq.formats.tpq import (
    TPQ_PQ_SPECS,
    TpqInt4Tensor,
    TpqPqTensor,
    normalize_tpq_dtype,
    pack_tpq_int4,
    pack_tpq_pq,
    unpack_tpq_int4,
    unpack_tpq_pq,
)
from mfq.quantize.nint_quant import NintTensor

MfqTensor: TypeAlias = (
    NintTensor
    | Nint8ZeroTensor
    | NintMoeTensor
    | NvqTensor
    | NvqJscTensor
    | Npq0LTensor
    | Npq0STensor
    | Nvq1LTensor
    | Nvq1STensor
    | NepqTensor
    | TpqPqTensor
    | TpqInt4Tensor
    | MxTensor
    | np.ndarray
    | bytes
)


class BFloat16Array(np.ndarray):
    """NumPy-compatible view of raw little-endian BF16 storage.

    NumPy does not provide a native BF16 dtype.  Keeping the payload as a
    tagged ``uint16`` ndarray preserves the exact on-disk bits while allowing
    the Torch and MLX runtimes to reinterpret them without a float32 staging
    copy.
    """


def is_bfloat16_array(value: object) -> bool:
    return isinstance(value, BFloat16Array)


def bfloat16_to_float32(value: np.ndarray) -> np.ndarray:
    """Decode a tagged/raw BF16 array into numerically exact float32 values."""

    bits = np.asarray(value, dtype="<u2")
    return (bits.astype(np.uint32) << 16).view(np.float32)


@dataclass(frozen=True)
class MMapTensorRecord:
    name: str
    dtype: str
    offset: int
    nbytes: int
    source_index: int = 0


def _u32(x: int) -> bytes:
    return struct.pack("<I", int(x))


def _read_u32(buf: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<I", buf, off)[0], off + 4


def _read_str(buf: bytes, off: int) -> tuple[str, int]:
    n, off = _read_u32(buf, off)
    s = buf[off : off + n].decode("utf-8")
    return s, off + n


# ---------------------------------------------------------------------------
# NINT codec packing / unpacking
# ---------------------------------------------------------------------------
_NINT_HDR = struct.Struct("<BBiii")   # bits, sub_bits, groupsize, axis, neuron_len


def pack_bits(values: np.ndarray, bits: int) -> bytes:
    """Pack unsigned integer values into a little-endian bitstream."""

    arr = np.ascontiguousarray(values).reshape(-1)
    if bits == 8:
        return arr.astype(np.uint8, copy=False).tobytes()
    if bits == 4:
        u = arr.astype(np.uint8, copy=False)
        if u.size % 2:
            u = np.concatenate([u, np.zeros(1, dtype=np.uint8)])
        packed = u[0::2] | (u[1::2] << 4)
        return np.ascontiguousarray(packed, dtype=np.uint8).tobytes()
    if bits < 8:
        u = arr.astype(np.uint8, copy=False)
        bit_rows = np.unpackbits(u[:, None], axis=1, bitorder="little")[:, :bits]
        return np.packbits(bit_rows.reshape(-1), bitorder="little").tobytes()
    maxval = (1 << bits) - 1
    udt = _uint_dtype(maxval)
    return arr.astype(udt, copy=False).tobytes()


def unpack_bits(blob: bytes, off: int, count: int, bits: int) -> tuple[np.ndarray, int]:
    """Unpack ``count`` unsigned integer values from a little-endian bitstream."""

    if bits == 8:
        end = off + count
        return np.frombuffer(blob, dtype=np.uint8, count=count, offset=off).copy(), end
    if bits == 4:
        nbytes = (count + 1) // 2
        packed = np.frombuffer(blob, dtype=np.uint8, count=nbytes, offset=off)
        out = np.empty(nbytes * 2, dtype=np.uint8)
        out[0::2] = packed & 0x0F
        out[1::2] = packed >> 4
        return out[:count].copy(), off + nbytes
    if bits < 8:
        nbytes = (count * bits + 7) // 8
        packed = np.frombuffer(blob, dtype=np.uint8, count=nbytes, offset=off)
        bitstream = np.unpackbits(packed, bitorder="little")[: count * bits].reshape(count, bits)
        shifts = (1 << np.arange(bits, dtype=np.uint16))
        out = (bitstream.astype(np.uint16) * shifts).sum(axis=1).astype(_uint_dtype((1 << bits) - 1))
        return out, off + nbytes
    maxval = (1 << bits) - 1
    udt = _uint_dtype(maxval)
    end = off + count * udt.itemsize
    return np.frombuffer(blob, dtype=udt, count=count, offset=off).copy(), end


def pack_nint(tensor: NintTensor) -> bytes:
    s = tensor.spec
    out, ng, _gs = tensor.q.shape
    parts = [_NINT_HDR.pack(s.bits, s.sub_bits, s.groupsize, tensor.axis, tensor.neuron_len)]
    parts.append(struct.pack("<I", len(tensor.shape)))
    parts.append(struct.pack(f"<{len(tensor.shape)}q", *tensor.shape))
    parts.append(struct.pack("<II", out, ng))
    parts.append(np.ascontiguousarray(tensor.neuron_scale, dtype=np.float16).tobytes())
    parts.append(np.ascontiguousarray(tensor.neuron_min, dtype=np.float16).tobytes())
    parts.append(pack_bits(tensor.sub_scale, s.sub_bits))
    parts.append(pack_bits(tensor.sub_min, s.sub_bits))
    parts.append(pack_bits(tensor.q, s.bits))
    return b"".join(parts)


def unpack_nint(blob: bytes) -> NintTensor:
    bits, sub_bits, groupsize, axis, neuron_len = _NINT_HDR.unpack_from(blob, 0)
    off = _NINT_HDR.size
    ndim = struct.unpack_from("<I", blob, off)[0]
    off += 4
    shape = struct.unpack_from(f"<{ndim}q", blob, off)
    off += 8 * ndim
    out, ng = struct.unpack_from("<II", blob, off)
    off += 8

    spec = NintSpec(bits=bits, groupsize=groupsize, sub_bits=sub_bits)
    neuron_scale = np.frombuffer(blob, dtype=np.float16, count=out, offset=off).astype(np.float32)
    off += out * 2
    neuron_min = np.frombuffer(blob, dtype=np.float16, count=out, offset=off).astype(np.float32)
    off += out * 2
    sub_count = out * ng
    q_count = out * ng * groupsize
    packed_tail = ((sub_count * sub_bits + 7) // 8) * 2 + (q_count * bits + 7) // 8
    old_sub_dtype = _uint_dtype((1 << sub_bits) - 1)
    old_q_dtype = _uint_dtype((1 << bits) - 1)
    old_tail = sub_count * old_sub_dtype.itemsize * 2 + q_count * old_q_dtype.itemsize
    remaining = len(blob) - off
    if remaining == old_tail:
        sub_scale = np.frombuffer(blob, dtype=old_sub_dtype, count=sub_count, offset=off).copy().reshape(out, ng)
        off += sub_count * old_sub_dtype.itemsize
        sub_min = np.frombuffer(blob, dtype=old_sub_dtype, count=sub_count, offset=off).copy().reshape(out, ng)
        off += sub_count * old_sub_dtype.itemsize
        q = np.frombuffer(blob, dtype=old_q_dtype, count=q_count, offset=off).copy()
    else:
        if remaining != packed_tail:
            raise ValueError(f"invalid NINT blob tail: remaining={remaining}, packed={packed_tail}, old={old_tail}")
        sub_scale, off = unpack_bits(blob, off, sub_count, sub_bits)
        sub_min, off = unpack_bits(blob, off, sub_count, sub_bits)
        q, off = unpack_bits(blob, off, q_count, bits)
        sub_scale = sub_scale.reshape(out, ng)
        sub_min = sub_min.reshape(out, ng)
    q = q.reshape(out, ng, groupsize)

    return NintTensor(
        spec=spec, shape=shape, axis=axis, q=q,
        neuron_scale=neuron_scale, neuron_min=neuron_min,
        sub_scale=sub_scale, sub_min=sub_min, neuron_len=neuron_len,
    )


_NINT_MOE_MAGIC_V1 = b"NIM1"
_NINT_MOE_MAGIC_V2 = b"NIM2"
_NINT_MOE_HDR = struct.Struct("<4sIIII")
_NINT_MOE_POOL_V1_HDR = struct.Struct("<IQ")
_NINT_MOE_POOL_V2_HDR = struct.Struct("<IIQQ")
_NINT_MOE_ROTATION_HDR = struct.Struct("<4sIIQ")
_NINT_MOE_ROTATION_MAGIC = b"HSG1"


def _pack_nint_moe_runtime(tensor: MfqTensor) -> bytes:
    if not isinstance(tensor, NepqTensor) or not tensor.rotation_block:
        return b""
    signs = rotation_signs(
        tensor.neuron_len, tensor.rotation_block, tensor.rotation_seed
    )
    return b"".join(
        [
            _NINT_MOE_ROTATION_HDR.pack(
                _NINT_MOE_ROTATION_MAGIC,
                int(tensor.neuron_len),
                int(tensor.rotation_block),
                int(tensor.rotation_seed),
            ),
            signs.tobytes(),
        ]
    )


def _validate_nint_moe_runtime(tensor: MfqTensor, payload: bytes) -> None:
    if not payload:
        if isinstance(tensor, NepqTensor) and tensor.rotation_block:
            raise ValueError("rotated NEPQ cohort lacks its runtime sign vector")
        return
    if not isinstance(tensor, NepqTensor) or not tensor.rotation_block:
        raise ValueError("unexpected NINTM cohort runtime metadata")
    if len(payload) < _NINT_MOE_ROTATION_HDR.size:
        raise ValueError("truncated NINTM rotation metadata")
    magic, width, block, seed = _NINT_MOE_ROTATION_HDR.unpack_from(payload)
    expected_size = _NINT_MOE_ROTATION_HDR.size + int(width)
    if (
        magic != _NINT_MOE_ROTATION_MAGIC
        or width != tensor.neuron_len
        or block != tensor.rotation_block
        or seed != tensor.rotation_seed
        or len(payload) != expected_size
    ):
        raise ValueError("NINTM rotation metadata does not match its NEPQ payload")
    expected = rotation_signs(width, block, seed)
    actual = np.frombuffer(payload, dtype=np.int8, offset=_NINT_MOE_ROTATION_HDR.size)
    if not np.array_equal(actual, expected):
        raise ValueError("NINTM rotation sign vector is corrupt")


def pack_nint_moe(tensor: NintMoeTensor) -> bytes:
    """Pack heterogeneous expert cohorts without changing local row order."""

    n_experts, out_per_expert, neuron_len = tensor.shape
    parts = [
        _NINT_MOE_HDR.pack(
            _NINT_MOE_MAGIC_V2,
            int(n_experts),
            int(out_per_expert),
            int(neuron_len),
            len(tensor.pools),
        )
    ]
    for pool in tensor.pools:
        expert_ids = np.ascontiguousarray(pool.expert_ids, dtype=np.int32).reshape(-1)
        dtype, payload = _pack_tensor(pool.tensor, allow_moe=False)
        runtime_payload = _pack_nint_moe_runtime(pool.tensor)
        dtype_bytes = dtype.encode("ascii")
        if not dtype_bytes or len(dtype_bytes) > 32:
            raise ValueError(f"invalid NINTM cohort dtype: {dtype!r}")
        parts.append(
            _NINT_MOE_POOL_V2_HDR.pack(
                expert_ids.size,
                len(dtype_bytes),
                len(payload),
                len(runtime_payload),
            )
        )
        parts.append(expert_ids.tobytes())
        parts.append(dtype_bytes)
        parts.append(runtime_payload)
        parts.append(payload)
    return b"".join(parts)


def _unpack_nint_moe_v1(
    blob: bytes | memoryview,
    *,
    n_experts: int,
    out_per_expert: int,
    neuron_len: int,
    pool_count: int,
) -> NintMoeTensor:
    off = _NINT_MOE_HDR.size
    pools: list[NintMoePool] = []
    for _ in range(pool_count):
        if off + _NINT_MOE_POOL_V1_HDR.size > len(blob):
            raise ValueError("truncated NINTM pool header")
        expert_count, payload_nbytes = _NINT_MOE_POOL_V1_HDR.unpack_from(blob, off)
        off += _NINT_MOE_POOL_V1_HDR.size
        ids_nbytes = int(expert_count) * np.dtype(np.int32).itemsize
        payload_end = off + ids_nbytes + int(payload_nbytes)
        if expert_count == 0 or payload_end > len(blob):
            raise ValueError("truncated NINTM pool payload")
        expert_ids = np.frombuffer(
            blob, dtype=np.int32, count=int(expert_count), offset=off
        ).copy()
        off += ids_nbytes
        tensor = unpack_nint(blob[off : off + int(payload_nbytes)])
        off += int(payload_nbytes)
        pools.append(NintMoePool(expert_ids=expert_ids, tensor=tensor))
    if off != len(blob):
        raise ValueError(f"invalid NINTM tail: {len(blob) - off} extra bytes")
    return NintMoeTensor(
        shape=(int(n_experts), int(out_per_expert), int(neuron_len)),
        pools=tuple(pools),
    )


def _unpack_nint_moe_v2(
    blob: bytes | memoryview,
    *,
    n_experts: int,
    out_per_expert: int,
    neuron_len: int,
    pool_count: int,
) -> NintMoeTensor:
    off = _NINT_MOE_HDR.size
    pools: list[NintMoePool] = []
    for _ in range(pool_count):
        if off + _NINT_MOE_POOL_V2_HDR.size > len(blob):
            raise ValueError("truncated NINTM v2 pool header")
        (
            expert_count,
            dtype_nbytes,
            payload_nbytes,
            runtime_nbytes,
        ) = _NINT_MOE_POOL_V2_HDR.unpack_from(blob, off)
        off += _NINT_MOE_POOL_V2_HDR.size
        if expert_count == 0 or dtype_nbytes == 0 or dtype_nbytes > 32:
            raise ValueError("invalid NINTM v2 pool metadata")
        ids_nbytes = int(expert_count) * np.dtype(np.int32).itemsize
        dtype_end = off + ids_nbytes + int(dtype_nbytes)
        runtime_end = dtype_end + int(runtime_nbytes)
        payload_end = runtime_end + int(payload_nbytes)
        if payload_end > len(blob):
            raise ValueError("truncated NINTM v2 pool payload")
        expert_ids = np.frombuffer(
            blob, dtype=np.int32, count=int(expert_count), offset=off
        ).copy()
        off += ids_nbytes
        try:
            dtype = bytes(blob[off : off + int(dtype_nbytes)]).decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("NINTM cohort dtype must be ASCII") from exc
        off += int(dtype_nbytes)
        runtime_payload = bytes(blob[off:runtime_end])
        off = runtime_end
        tensor = _unpack_tensor(dtype, blob[off:payload_end])
        if isinstance(tensor, (NintMoeTensor, np.ndarray)):
            raise ValueError(f"unsupported nested NINTM cohort dtype: {dtype}")
        _validate_nint_moe_runtime(tensor, runtime_payload)
        off = payload_end
        pools.append(NintMoePool(expert_ids=expert_ids, tensor=tensor))
    if off != len(blob):
        raise ValueError(f"invalid NINTM tail: {len(blob) - off} extra bytes")
    return NintMoeTensor(
        shape=(int(n_experts), int(out_per_expert), int(neuron_len)),
        pools=tuple(pools),
    )


def unpack_nint_moe(blob: bytes | memoryview) -> NintMoeTensor:
    """Decode a versioned ``NINTM`` expert-wise precision container."""

    if len(blob) < _NINT_MOE_HDR.size:
        raise ValueError("truncated NINTM header")
    magic, n_experts, out_per_expert, neuron_len, pool_count = _NINT_MOE_HDR.unpack_from(
        blob, 0
    )
    if pool_count == 0 or pool_count > n_experts:
        raise ValueError("invalid NINTM pool count")
    kwargs = {
        "n_experts": int(n_experts),
        "out_per_expert": int(out_per_expert),
        "neuron_len": int(neuron_len),
        "pool_count": int(pool_count),
    }
    if magic == _NINT_MOE_MAGIC_V1:
        return _unpack_nint_moe_v1(blob, **kwargs)
    if magic == _NINT_MOE_MAGIC_V2:
        return _unpack_nint_moe_v2(blob, **kwargs)
    raise ValueError(f"invalid NINTM magic: {magic!r}")


_DENSE_DTYPES = {
    "BF16": np.dtype("<u2"),
    "F16": np.dtype(np.float16),
    "F32": np.dtype(np.float32),
    "I32": np.dtype(np.int32),
    "I64": np.dtype(np.int64),
}
_DENSE_NAMES = {
    value: name for name, value in _DENSE_DTYPES.items() if name != "BF16"
}


def pack_dense(tensor: np.ndarray) -> tuple[str, bytes]:
    """Pack a small dense tensor, used for norm weights and metadata-like arrays."""

    bfloat16 = is_bfloat16_array(tensor)
    arr = np.ascontiguousarray(tensor)
    dtype = "BF16" if bfloat16 else _DENSE_NAMES.get(arr.dtype)
    if dtype is None:
        raise ValueError(f"unsupported dense dtype: {arr.dtype}")
    parts = [struct.pack("<I", arr.ndim)]
    parts.append(struct.pack(f"<{arr.ndim}q", *arr.shape))
    parts.append(arr.tobytes())
    return dtype, b"".join(parts)


def unpack_dense(dtype: str, blob: bytes) -> np.ndarray:
    if dtype not in _DENSE_DTYPES:
        raise ValueError(f"unknown dense dtype: {dtype}")
    off = 0
    ndim = struct.unpack_from("<I", blob, off)[0]
    off += 4
    shape = struct.unpack_from(f"<{ndim}q", blob, off)
    off += 8 * ndim
    arr = np.frombuffer(blob, dtype=_DENSE_DTYPES[dtype], offset=off).copy()
    result = arr.reshape(shape)
    return result.view(BFloat16Array) if dtype == "BF16" else result


def _unpack_tensor(dtype: str, blob: bytes | memoryview) -> MfqTensor:
    if dtype == ASSET_DTYPE:
        return bytes(blob)
    if dtype in MX_DTYPES:
        return unpack_mx(dtype, blob)
    dtype = {
        "NIQ2": "NVQ2",
        "NIQ2J": "NVQ2J",
        "NIQ3": "NVQ3",
    }.get(dtype, dtype)
    dtype = normalize_tpq_dtype(dtype)
    if dtype == "NINTM":
        return unpack_nint_moe(blob)
    if dtype == "NINT8-0":
        return unpack_nint8_zero(blob)
    if dtype == "TPQ-I4G64":
        tensor = unpack_tpq_int4(blob)
        if tensor.group_size != 64:
            raise ValueError(
                f"MFQ dtype/blob mismatch: TPQ-I4G64 contains g{tensor.group_size}"
            )
        return tensor
    if dtype in TPQ_PQ_SPECS or dtype == "TPQ-P":
        tensor = unpack_tpq_pq(blob)
        if tensor.spec.label != dtype:
            raise ValueError(
                f"MFQ dtype/blob mismatch: {dtype} contains {tensor.spec.label}"
            )
        return tensor
    if dtype in {
        "NEPQ0-S",
        "NEPQ0-L",
        "NEPQ1-S",
        "NEPQ1-L",
        "NEPQ0-A",
        "NEPQ1-A",
    }:
        tensor = unpack_nepq(blob)
        if tensor.spec.label != dtype:
            raise ValueError(
                f"MFQ dtype/blob mismatch: {dtype} contains {tensor.spec.label}"
            )
        return tensor
    if dtype.startswith("NINT"):
        return unpack_nint(blob)
    if dtype == "NPQ0-L":
        return unpack_npq0_l(blob)
    if dtype == "NPQ0-S":
        return unpack_npq0_s(blob)
    if dtype == "NVQ1-L":
        return unpack_nvq1_l(blob)
    if dtype == "NVQ1-S":
        return unpack_nvq1_s(blob)
    if dtype in {
        "NVQ2",
        "NVQ2J",
        "NVQ2J-L",
        "NVQ2J-XL",
        "NVQ3",
        "NVQ3J",
        "NVQ3J-512",
        "NVQ3J-L",
    }:
        tensor = unpack_nvq(blob)
        if dtype in {
            "NVQ2J",
            "NVQ2J-L",
            "NVQ2J-XL",
            "NVQ3J",
            "NVQ3J-512",
            "NVQ3J-L",
        }:
            if not isinstance(tensor, NvqJscTensor):
                raise ValueError(f"MFQ dtype/blob mismatch: {dtype} lacks the JSC profile")
            expected = {
                "NVQ2J": "e8_256",
                "NVQ2J-L": "e8_1024",
                "NVQ2J-XL": "e8_4096",
                "NVQ3J": "d4_256",
                "NVQ3J-512": "d4_512",
                "NVQ3J-L": "d4_1024",
            }[dtype]
            if tensor.spec.codebook != expected:
                raise ValueError(
                    f"MFQ dtype/blob mismatch: {dtype} contains {tensor.spec.codebook}"
                )
            return tensor
        if isinstance(tensor, NvqJscTensor):
            raise ValueError(f"MFQ dtype/blob mismatch: {dtype} contains NVQ-JSC")
        expected = "e8_256" if dtype == "NVQ2" else "d4_256"
        if tensor.spec.codebook != expected:
            raise ValueError(
                f"MFQ dtype/blob mismatch: {dtype} contains {tensor.spec.codebook}"
            )
        return tensor
    return unpack_dense(dtype, blob)


def unpack_tensor_payload(dtype: str, blob: bytes | memoryview) -> MfqTensor:
    """Decode one self-contained tensor payload using its public MFQ dtype."""

    return _unpack_tensor(dtype, blob)


def _pack_tensor(tensor: MfqTensor, *, allow_moe: bool = True) -> tuple[str, bytes]:
    """Return one tensor's public dtype label and native payload."""

    if isinstance(tensor, bytes):
        return ASSET_DTYPE, tensor
    if isinstance(tensor, MxTensor):
        return tensor.dtype, pack_mx(tensor)
    if isinstance(tensor, Nint8ZeroTensor):
        return "NINT8-0", pack_nint8_zero(tensor)
    if isinstance(tensor, TpqInt4Tensor):
        if tensor.group_size != 64:
            raise ValueError(
                f"MFQ only names the production TPQ int4-g64 profile, got g{tensor.group_size}"
            )
        return "TPQ-I4G64", pack_tpq_int4(tensor)
    if isinstance(tensor, TpqPqTensor):
        return tensor.spec.label, pack_tpq_pq(tensor)
    if isinstance(tensor, NintTensor):
        return f"NINT{tensor.spec.bits}", pack_nint(tensor)
    if isinstance(tensor, NintMoeTensor):
        if not allow_moe:
            raise TypeError("nested NINTM cohorts are not supported")
        return "NINTM", pack_nint_moe(tensor)
    if isinstance(tensor, NepqTensor):
        return tensor.spec.label, pack_nepq(tensor)
    if isinstance(tensor, Nvq1LTensor):
        return "NVQ1-L", pack_nvq1_l(tensor)
    if isinstance(tensor, Nvq1STensor):
        return "NVQ1-S", pack_nvq1_s(tensor)
    if isinstance(tensor, Npq0LTensor):
        return "NPQ0-L", pack_npq0_l(tensor)
    if isinstance(tensor, Npq0STensor):
        return "NPQ0-S", pack_npq0_s(tensor)
    if isinstance(tensor, NvqJscTensor):
        dtype = {
            "e8_256": "NVQ2J",
            "e8_1024": "NVQ2J-L",
            "e8_4096": "NVQ2J-XL",
            "d4_256": "NVQ3J",
            "d4_512": "NVQ3J-512",
            "d4_1024": "NVQ3J-L",
        }[tensor.spec.codebook]
        return dtype, pack_nvq(tensor)
    if isinstance(tensor, NvqTensor):
        dtype = {
            "e8_256": "NVQ2",
            "d4_256": "NVQ3",
        }.get(tensor.spec.codebook)
        if dtype is None:
            raise ValueError(
                f"{tensor.spec.codebook} requires an NvqJscTensor file profile"
            )
        return dtype, pack_nvq(tensor)
    if isinstance(tensor, np.ndarray):
        return pack_dense(tensor)
    raise TypeError(f"unsupported tensor type: {type(tensor)!r}")


# ---------------------------------------------------------------------------
# File-level save / load
# ---------------------------------------------------------------------------
def save(path: str | Path, header: FileHeader, tensors: dict[str, MfqTensor]) -> None:
    """Write an ``.mfq`` file."""

    packed: dict[str, tuple[str, bytes]] = {}
    for name, t in tensors.items():
        try:
            packed[name] = _pack_tensor(t)
        except TypeError as exc:
            raise TypeError(f"unsupported tensor type for {name!r}: {type(t)!r}") from exc
    version = int(header.version)
    extra = dict(header.extra)
    if extra and version < 2:
        version = 2
    with open(path, "wb") as f:
        f.write(MFQ_MAGIC)
        f.write(_u32(version))
        arch_b = header.model_arch.encode("utf-8")
        f.write(_u32(len(arch_b)))
        f.write(arch_b)
        if version >= 2:
            f.write(_u32(len(extra)))
            for k, v in extra.items():
                kb = str(k).encode("utf-8")
                vb = json.dumps(v).encode("utf-8")
                f.write(_u32(len(kb)))
                f.write(kb)
                f.write(_u32(len(vb)))
                f.write(vb)
        f.write(_u32(len(tensors)))
        for name in tensors:
            name_b = name.encode("utf-8")
            dtype, blob = packed[name]
            dtype_b = dtype.encode("utf-8")
            f.write(_u32(len(name_b)))
            f.write(name_b)
            f.write(_u32(len(dtype_b)))
            f.write(dtype_b)
            f.write(struct.pack("<Q", len(blob)))
        for name in tensors:
            f.write(packed[name][1])


def load(path: str | Path) -> tuple[FileHeader, dict[str, MfqTensor]]:
    """Read an ``.mfq`` file and return ``(header, {name: NintTensor})``."""

    with open_mmap(path) as store:
        return store.header, {name: store[name] for name in store}


class MMapTensorStore(Mapping[str, MfqTensor]):
    """Lazy mmap-backed MFQ tensor store.

    The file header and tensor table are parsed eagerly. Tensor blobs are decoded
    only when ``__getitem__`` is called. By default decoded tensors are not cached,
    so model construction can load one tensor, move it to GPU, and let CPU memory
    be reclaimed before the next tensor.
    """

    def __init__(
        self,
        path: str | Path,
        header: FileHeader,
        records: dict[str, MMapTensorRecord],
        file_obj,
        mm: mmap.mmap,
        *,
        cache: bool = False,
        paths: list[Path] | None = None,
        file_objs: list[object] | None = None,
        mmaps: list[mmap.mmap] | None = None,
    ) -> None:
        self.path = Path(path)
        self.header = header
        self.records = records
        self.paths = list(paths or [self.path])
        self._files = list(file_objs or [file_obj])
        self._mmaps = list(mmaps or [mm])
        # Kept for single-file callers; shard-aware code uses the helpers below.
        self._file = self._files[0]
        self._mmap = self._mmaps[0]
        self._cache_enabled = cache
        self._cache: dict[str, MfqTensor] = {}

    def mmap_for(self, record: MMapTensorRecord | str) -> mmap.mmap:
        rec = self.records[record] if isinstance(record, str) else record
        return self._mmaps[rec.source_index]

    def file_for(self, record: MMapTensorRecord | str):
        rec = self.records[record] if isinstance(record, str) else record
        return self._files[rec.source_index]

    def blob_view(self, record: MMapTensorRecord | str) -> memoryview:
        rec = self.records[record] if isinstance(record, str) else record
        return memoryview(self.mmap_for(rec))[rec.offset : rec.offset + rec.nbytes]

    def __getitem__(self, name: str) -> MfqTensor:
        if self._cache_enabled and name in self._cache:
            return self._cache[name]
        rec = self.records[name]
        blob = self.blob_view(rec)
        try:
            tensor = _unpack_tensor(rec.dtype, blob)
        finally:
            blob.release()
        if self._cache_enabled:
            self._cache[name] = tensor
        return tensor

    def __iter__(self) -> Iterator[str]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def read_blob(self, name: str) -> bytes:
        """Copy one tensor's original packed payload without decoding it."""

        rec = self.records[name]
        return bytes(
            self.mmap_for(rec)[rec.offset : rec.offset + rec.nbytes]
        )

    def close(self) -> None:
        self._cache.clear()
        for mm in self._mmaps:
            mm.close()
        for file_obj in self._files:
            file_obj.close()

    def __enter__(self) -> MMapTensorStore:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _open_single_mmap(
    path: str | Path,
    *,
    source_index: int = 0,
) -> tuple[FileHeader, dict[str, MMapTensorRecord], object, mmap.mmap]:
    p = Path(path)
    f = p.open("rb")
    mm = None
    try:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        if mm[:4] != MFQ_MAGIC:
            raise ValueError(f"非 MFQ 文件（魔数不匹配）: {mm[:4]!r}")
        off = 4
        version, off = _read_u32(mm, off)
        arch, off = _read_str(mm, off)
        extra: dict[str, str] = {}
        if version >= 2:
            n_extra, off = _read_u32(mm, off)
            for _ in range(n_extra):
                k, off = _read_str(mm, off)
                v_json, off = _read_str(mm, off)
                extra[k] = json.loads(v_json)
        num_tensors, off = _read_u32(mm, off)

        raw_records: list[tuple[str, str, int]] = []
        for _ in range(num_tensors):
            name, off = _read_str(mm, off)
            dtype, off = _read_str(mm, off)
            blob_nbytes = struct.unpack_from("<Q", mm, off)[0]
            off += 8
            raw_records.append((name, dtype, blob_nbytes))

        records: dict[str, MMapTensorRecord] = {}
        blob_off = off
        for name, dtype, blob_nbytes in raw_records:
            if name in records:
                raise ValueError(f"duplicate MFQ tensor record: {name}: {p}")
            records[name] = MMapTensorRecord(
                name, dtype, blob_off, blob_nbytes, source_index
            )
            blob_off += blob_nbytes
        if blob_off != mm.size():
            raise ValueError(f"MFQ 文件长度不匹配: records end={blob_off}, file size={mm.size()}")
        header = FileHeader(version=version, model_arch=arch, num_tensors=num_tensors, extra=extra)
        return header, records, f, mm
    except Exception:
        if mm is not None:
            mm.close()
        f.close()
        raise


def open_mmap(path: str | Path, *, cache: bool = False) -> MMapTensorStore:
    """Open a single-file or sharded MFQ as one lazy tensor mapping."""

    from mfq.formats.assets import is_asset_record
    from mfq.formats.shards import (
        parse_shard_path,
        shard_paths_from_any,
        split_values,
    )

    requested = Path(path)
    first_header, first_records, first_file, first_mmap = _open_single_mmap(requested)
    opened_files = [first_file]
    opened_mmaps = [first_mmap]
    try:
        split_no, split_count, expected_tensors, expected_records = split_values(
            first_header.extra
        )
        parsed = parse_shard_path(requested)
        if split_count == 1:
            if parsed is not None and parsed[2] != 1:
                raise ValueError(
                    f"MFQ filename claims multiple shards but metadata does not: {requested}"
                )
            return MMapTensorStore(
                requested,
                first_header,
                first_records,
                first_file,
                first_mmap,
                cache=cache,
            )
        if parsed is None:
            raise ValueError(
                f"sharded MFQ path lacks -00001-of-00000 suffix: {requested}"
            )
        if parsed[1] - 1 != split_no:
            raise ValueError(
                f"MFQ shard index mismatch in filename/metadata: "
                f"{parsed[1] - 1} != {split_no}: {requested}"
            )
        if split_no:
            first_records = {
                name: MMapTensorRecord(
                    record.name,
                    record.dtype,
                    record.offset,
                    record.nbytes,
                    split_no,
                )
                for name, record in first_records.items()
            }
        paths = shard_paths_from_any(requested, split_count)
        missing = [value for value in paths if not value.is_file()]
        if missing:
            raise FileNotFoundError(f"missing MFQ shard: {missing[0]}")

        headers: list[FileHeader | None] = [None] * split_count
        shard_records: list[dict[str, MMapTensorRecord] | None] = [None] * split_count
        files: list[object | None] = [None] * split_count
        mmaps: list[mmap.mmap | None] = [None] * split_count
        headers[split_no] = first_header
        shard_records[split_no] = first_records
        files[split_no] = first_file
        mmaps[split_no] = first_mmap

        for index, shard_path in enumerate(paths):
            if index == split_no:
                continue
            header, records, file_obj, mm = _open_single_mmap(
                shard_path, source_index=index
            )
            opened_files.append(file_obj)
            opened_mmaps.append(mm)
            headers[index] = header
            shard_records[index] = records
            files[index] = file_obj
            mmaps[index] = mm

        combined: dict[str, MMapTensorRecord] = {}
        total_tensors = 0
        reference_version = first_header.version
        reference_arch = first_header.model_arch
        for index in range(split_count):
            header = headers[index]
            records = shard_records[index]
            assert header is not None and records is not None
            no, count, tensors, record_count = split_values(header.extra)
            if no != index or count != split_count:
                raise ValueError(
                    f"MFQ shard metadata mismatch: expected {index}/{split_count}, "
                    f"got {no}/{count}: {paths[index]}"
                )
            if header.version != reference_version or header.model_arch != reference_arch:
                raise ValueError(f"MFQ shard architecture/version mismatch: {paths[index]}")
            if expected_tensors is None:
                expected_tensors = tensors
            elif tensors is not None and tensors != expected_tensors:
                raise ValueError(f"MFQ shard tensor count mismatch: {paths[index]}")
            if expected_records is None:
                expected_records = record_count
            elif record_count is not None and record_count != expected_records:
                raise ValueError(f"MFQ shard record count mismatch: {paths[index]}")
            for name, record in records.items():
                if name in combined:
                    raise ValueError(f"duplicate MFQ tensor across shards: {name}")
                combined[name] = record
                total_tensors += not is_asset_record(name)

        if expected_records is not None and len(combined) != expected_records:
            raise ValueError(
                f"MFQ shard record total mismatch: {len(combined)} != {expected_records}"
            )
        if expected_tensors is not None and total_tensors != expected_tensors:
            raise ValueError(
                f"MFQ shard tensor total mismatch: {total_tensors} != {expected_tensors}"
            )
        primary = headers[0]
        assert primary is not None
        primary.num_tensors = len(combined)
        return MMapTensorStore(
            paths[0],
            primary,
            combined,
            files[0],
            mmaps[0],
            cache=cache,
            paths=paths,
            file_objs=[value for value in files if value is not None],
            mmaps=[value for value in mmaps if value is not None],
        )
    except Exception:
        for mm in opened_mmaps:
            if not mm.closed:
                mm.close()
        for file_obj in opened_files:
            if not file_obj.closed:
                file_obj.close()
        raise


def load_mmap(path: str | Path, *, cache: bool = False) -> tuple[FileHeader, MMapTensorStore]:
    """Return ``(header, lazy mmap tensor store)`` without reading all blobs."""

    store = open_mmap(path, cache=cache)
    return store.header, store
