"""Streaming readers and GPU decoders for MXFP4/MXFP8 safetensors.

PyTorch 2.6 cannot materialize ``F8_E8M0`` safetensors.  The reader below
therefore maps tensor payload bytes directly and transfers only requested
rows to the target device.  MX arithmetic stays on the target device.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


_FP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_DTYPE_NBYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_FP4_TABLES: dict[str, torch.Tensor] = {}
_E4M3_TABLES: dict[str, torch.Tensor] = {}


@dataclass(frozen=True)
class SafeTensorInfo:
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int

    @property
    def nbytes(self) -> int:
        return self.data_end - self.data_start


class RawSafeTensorFile:
    """Read contiguous row ranges without materializing a safetensors shard."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        with self.path.open("rb") as handle:
            header_nbytes = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_nbytes))
        self.data_offset = 8 + int(header_nbytes)
        self._tensors: dict[str, SafeTensorInfo] = {}
        for name, raw in header.items():
            if name == "__metadata__":
                continue
            dtype = str(raw["dtype"])
            shape = tuple(int(value) for value in raw["shape"])
            begin, end = (int(value) for value in raw["data_offsets"])
            if dtype not in _DTYPE_NBYTES:
                raise ValueError(f"unsupported safetensors dtype {dtype}: {self.path}")
            expected = math.prod(shape) * _DTYPE_NBYTES[dtype]
            if end - begin != expected:
                raise ValueError(
                    f"safetensors byte count mismatch for {name}: "
                    f"{end - begin} != {expected}"
                )
            self._tensors[str(name)] = SafeTensorInfo(
                dtype=dtype,
                shape=shape,
                data_start=self.data_offset + begin,
                data_end=self.data_offset + end,
            )

    def info(self, name: str) -> SafeTensorInfo:
        try:
            return self._tensors[name]
        except KeyError as exc:
            raise KeyError(f"{name!r} is absent from {self.path}") from exc

    def raw_rows(self, name: str, start: int, stop: int) -> np.memmap:
        info = self.info(name)
        if len(info.shape) < 1:
            raise ValueError(f"cannot row-slice scalar tensor {name}")
        rows = info.shape[0]
        if start < 0 or stop < start or stop > rows:
            raise IndexError(f"invalid row slice {start}:{stop} for {name} {info.shape}")
        row_values = math.prod(info.shape[1:])
        row_bytes = row_values * _DTYPE_NBYTES[info.dtype]
        return np.memmap(
            self.path,
            mode="c",
            dtype=np.uint8,
            offset=info.data_start + start * row_bytes,
            shape=(stop - start, row_bytes),
            order="C",
        )

    def raw_tensor(self, name: str) -> np.memmap:
        info = self.info(name)
        return np.memmap(
            self.path,
            mode="c",
            dtype=np.uint8,
            offset=info.data_start,
            shape=(info.nbytes,),
            order="C",
        )


def _device_key(device: str | torch.device) -> str:
    return str(torch.device(device))


def _fp4_table(device: str | torch.device) -> torch.Tensor:
    key = _device_key(device)
    table = _FP4_TABLES.get(key)
    if table is None:
        table = torch.tensor(_FP4_VALUES, device=device, dtype=torch.float32)
        _FP4_TABLES[key] = table
    return table


def _e4m3_table(device: str | torch.device) -> torch.Tensor:
    key = _device_key(device)
    table = _E4M3_TABLES.get(key)
    if table is None:
        raw = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn)
        table = raw.to(torch.float32).to(device)
        _E4M3_TABLES[key] = table
    return table


def _u8_tensor(
    value: np.ndarray | torch.Tensor,
    device: str | torch.device,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.uint8, non_blocking=True)
    return torch.from_numpy(np.asarray(value, dtype=np.uint8)).to(
        device=device, non_blocking=True
    )


def decode_e8m0(
    value: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device,
) -> torch.Tensor:
    """Decode E8M0FNU bytes; 0..254 map to ``2**(byte-127)``."""

    raw = _u8_tensor(value, device)
    exponent = raw.to(torch.int32) - 127
    decoded = torch.ldexp(torch.ones_like(raw, dtype=torch.float32), exponent)
    return torch.where(raw == 255, torch.full_like(decoded, torch.nan), decoded)


def decode_mxfp4(
    packed: np.ndarray | torch.Tensor,
    scale: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode low-nibble-first E2M1 values with one E8M0 scale per 32 values."""

    raw = _u8_tensor(packed, device)
    if raw.ndim != 2:
        raise ValueError("MXFP4 packed values must be a rank-2 array")
    rows, packed_width = raw.shape
    width = packed_width * 2
    scale_raw = _u8_tensor(scale, device)
    if tuple(scale_raw.shape) != (rows, width // 32):
        raise ValueError(
            f"MXFP4 scale shape {tuple(scale_raw.shape)} does not match "
            f"packed shape {tuple(raw.shape)}"
        )
    table = _fp4_table(device)
    result = torch.empty((rows, width), device=device, dtype=torch.float32)
    result[:, 0::2] = table[(raw & 0x0F).to(torch.int64)]
    result[:, 1::2] = table[((raw >> 4) & 0x0F).to(torch.int64)]
    result.mul_(decode_e8m0(scale_raw, device=device).repeat_interleave(32, dim=1))
    return result.to(dtype=dtype)


def decode_mxfp8(
    encoded: np.ndarray | torch.Tensor,
    block_scale: np.ndarray | torch.Tensor,
    *,
    row_start: int,
    total_rows: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode E4M3 values with one E8M0 scale per 128x128 block."""

    raw = _u8_tensor(encoded, device)
    if raw.ndim != 2 or raw.shape[1] % 128:
        raise ValueError("MXFP8 encoded values must have shape [rows,K], K%128=0")
    rows, width = raw.shape
    if row_start < 0 or row_start + rows > total_rows:
        raise IndexError("MXFP8 row range is outside the tensor")
    first_block = row_start // 128
    last_block = (row_start + rows - 1) // 128 if rows else first_block - 1
    expected_scale_rows = max(0, last_block - first_block + 1)
    scale_raw = _u8_tensor(block_scale, device)
    if tuple(scale_raw.shape) != (expected_scale_rows, width // 128):
        raise ValueError(
            f"MXFP8 scale shape {tuple(scale_raw.shape)} does not match "
            f"rows={row_start}:{row_start + rows}, width={width}"
        )
    if rows == 0:
        return torch.empty((0, width), device=device, dtype=dtype)
    local_block = (
        torch.arange(row_start, row_start + rows, device=device, dtype=torch.int64)
        // 128
        - first_block
    )
    row_scale = decode_e8m0(scale_raw, device=device)[local_block]
    row_scale = row_scale.repeat_interleave(128, dim=1)
    result = _e4m3_table(device)[raw.to(torch.int64)] * row_scale
    return result.to(dtype=dtype)


def read_mxfp4_rows(
    shard: RawSafeTensorFile,
    weight_name: str,
    scale_name: str,
    start: int,
    stop: int,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    weight = shard.info(weight_name)
    scales = shard.info(scale_name)
    if weight.dtype != "I8" or scales.dtype != "F8_E8M0":
        raise ValueError(
            f"{weight_name} is not packed MXFP4: {weight.dtype}/{scales.dtype}"
        )
    packed = shard.raw_rows(weight_name, start, stop)
    scale = shard.raw_rows(scale_name, start, stop)
    return decode_mxfp4(
        packed,
        scale,
        device=device,
        dtype=dtype,
    )


def read_mxfp8_rows(
    shard: RawSafeTensorFile,
    weight_name: str,
    scale_name: str,
    start: int,
    stop: int,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    weight = shard.info(weight_name)
    scales = shard.info(scale_name)
    if weight.dtype not in {"F8_E4M3", "F8_E4M3FN"} or scales.dtype != "F8_E8M0":
        raise ValueError(
            f"{weight_name} is not MXFP8: {weight.dtype}/{scales.dtype}"
        )
    if len(weight.shape) != 2 or len(scales.shape) != 2:
        raise ValueError("MXFP8 reader accepts rank-2 tensors only")
    first_block = start // 128
    last_block = (stop - 1) // 128 if stop else first_block - 1
    encoded = shard.raw_rows(weight_name, start, stop)
    block_scale = shard.raw_rows(scale_name, first_block, last_block + 1)
    return decode_mxfp8(
        encoded,
        block_scale,
        row_start=start,
        total_rows=weight.shape[0],
        device=device,
        dtype=dtype,
    )


def read_dense_rows(
    shard: RawSafeTensorFile,
    name: str,
    start: int,
    stop: int,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None = torch.float32,
) -> torch.Tensor:
    """Read ordinary BF16/F16/F32/I32/I64 rows."""

    info = shard.info(name)
    raw = shard.raw_rows(name, start, stop)
    shape = (stop - start, *info.shape[1:])
    if info.dtype == "BF16":
        words = np.asarray(raw).view("<u2").reshape(shape)
        fp32 = (words.astype(np.uint32) << 16).view(np.float32)
        value = torch.from_numpy(fp32)
    elif info.dtype == "F16":
        value = torch.from_numpy(np.asarray(raw).view("<f2").reshape(shape))
    elif info.dtype == "F32":
        value = torch.from_numpy(np.asarray(raw).view("<f4").reshape(shape))
    elif info.dtype == "I32":
        value = torch.from_numpy(np.asarray(raw).view("<i4").reshape(shape))
    elif info.dtype == "I64":
        value = torch.from_numpy(np.asarray(raw).view("<i8").reshape(shape))
    else:
        raise ValueError(f"unsupported dense source dtype: {info.dtype}")
    return value.to(
        device=device,
        dtype=value.dtype if dtype is None else dtype,
        non_blocking=True,
    )


__all__ = [
    "RawSafeTensorFile",
    "SafeTensorInfo",
    "decode_e8m0",
    "decode_mxfp4",
    "decode_mxfp8",
    "read_dense_rows",
    "read_mxfp4_rows",
    "read_mxfp8_rows",
]
