"""Self-contained OCP MX tensor payloads used by full-precision MFQ files.

The official DeepSeek-V4-Flash checkpoint stores a logical matrix as two
safetensors entries: encoded values and E8M0 block scales.  MFQ keeps those
two byte streams in one tensor record so a weight can never be separated from
its scale tensor during sharding or conversion.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MXFP4_DTYPE = "MXFP4"
MXFP8_DTYPE = "MXFP8"
MX_DTYPES = frozenset({MXFP4_DTYPE, MXFP8_DTYPE})

_MAGIC = b"MXT1"
_VERSION = 1
_KIND_MXFP4 = 4
_KIND_MXFP8 = 8
_HEADER = struct.Struct("<4sBBHQQQQQQ")


@dataclass(frozen=True)
class MxTensorLayout:
    """Validated offsets and shapes for one packed MX payload."""

    dtype: str
    shape: tuple[int, int]
    storage_shape: tuple[int, int]
    scale_shape: tuple[int, int]
    values_offset: int
    values_nbytes: int
    scales_offset: int
    scales_nbytes: int


@dataclass(frozen=True)
class MxTensor:
    """An encoded MX matrix with raw uint8 values and E8M0 scales."""

    dtype: str
    shape: tuple[int, int]
    values: np.ndarray
    scales: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        scales = np.asarray(self.scales)
        layout = validate_mx_shapes(
            self.dtype,
            self.shape,
            tuple(int(value) for value in values.shape),
            tuple(int(value) for value in scales.shape),
        )
        if values.dtype.itemsize != 1 or scales.dtype.itemsize != 1:
            raise ValueError("MX values and E8M0 scales must be byte arrays")
        if layout.values_nbytes != values.nbytes:
            raise ValueError("MX values are not a contiguous byte matrix")


def validate_mx_shapes(
    dtype: str,
    shape: tuple[int, int],
    storage_shape: tuple[int, int],
    scale_shape: tuple[int, int],
) -> MxTensorLayout:
    """Validate OCP block geometry and return its payload layout."""

    if dtype not in MX_DTYPES:
        raise ValueError(f"unsupported MX dtype: {dtype}")
    if len(shape) != 2 or len(storage_shape) != 2 or len(scale_shape) != 2:
        raise ValueError("MFQ MX tensors must be rank-2 matrices")
    rows, columns = (int(value) for value in shape)
    storage_rows, storage_columns = (int(value) for value in storage_shape)
    scale_rows, scale_columns = (int(value) for value in scale_shape)
    if rows <= 0 or columns <= 0:
        raise ValueError(f"invalid MX logical shape: {shape}")
    if dtype == MXFP4_DTYPE:
        if columns % 32:
            raise ValueError("MXFP4 columns must be divisible by 32")
        expected_storage = (rows, columns // 2)
        expected_scales = (rows, columns // 32)
    else:
        if columns % 128:
            raise ValueError("MXFP8 columns must be divisible by 128")
        expected_storage = (rows, columns)
        expected_scales = ((rows + 127) // 128, columns // 128)
    if (storage_rows, storage_columns) != expected_storage:
        raise ValueError(
            f"{dtype} storage shape {storage_shape} != {expected_storage}"
        )
    if (scale_rows, scale_columns) != expected_scales:
        raise ValueError(f"{dtype} scale shape {scale_shape} != {expected_scales}")
    values_nbytes = storage_rows * storage_columns
    scales_nbytes = scale_rows * scale_columns
    return MxTensorLayout(
        dtype=dtype,
        shape=(rows, columns),
        storage_shape=(storage_rows, storage_columns),
        scale_shape=(scale_rows, scale_columns),
        values_offset=_HEADER.size,
        values_nbytes=values_nbytes,
        scales_offset=_HEADER.size + values_nbytes,
        scales_nbytes=scales_nbytes,
    )


def pack_mx(tensor: MxTensor) -> bytes:
    values = np.ascontiguousarray(tensor.values).view(np.uint8)
    scales = np.ascontiguousarray(tensor.scales).view(np.uint8)
    layout = validate_mx_shapes(
        tensor.dtype,
        tensor.shape,
        tuple(int(value) for value in values.shape),
        tuple(int(value) for value in scales.shape),
    )
    header = mx_header_bytes(
        tensor.dtype,
        layout.shape,
        layout.storage_shape,
        layout.scale_shape,
    )
    return header + values.tobytes() + scales.tobytes()


def mx_header_bytes(
    dtype: str,
    shape: tuple[int, int],
    storage_shape: tuple[int, int],
    scale_shape: tuple[int, int],
) -> bytes:
    """Build a validated header for a streamed MX payload writer."""

    layout = validate_mx_shapes(dtype, shape, storage_shape, scale_shape)
    kind = _KIND_MXFP4 if dtype == MXFP4_DTYPE else _KIND_MXFP8
    return _HEADER.pack(
        _MAGIC,
        _VERSION,
        kind,
        0,
        *layout.shape,
        *layout.storage_shape,
        *layout.scale_shape,
    )


def parse_mx_layout(dtype: str, blob: bytes | memoryview) -> MxTensorLayout:
    if len(blob) < _HEADER.size:
        raise ValueError("truncated MFQ MX tensor header")
    magic, version, kind, reserved, *dimensions = _HEADER.unpack_from(blob)
    expected_kind = _KIND_MXFP4 if dtype == MXFP4_DTYPE else _KIND_MXFP8
    if magic != _MAGIC or version != _VERSION or kind != expected_kind or reserved:
        raise ValueError(
            f"invalid {dtype} payload header: "
            f"magic={magic!r}, version={version}, kind={kind}"
        )
    layout = validate_mx_shapes(
        dtype,
        (int(dimensions[0]), int(dimensions[1])),
        (int(dimensions[2]), int(dimensions[3])),
        (int(dimensions[4]), int(dimensions[5])),
    )
    expected = _HEADER.size + layout.values_nbytes + layout.scales_nbytes
    if len(blob) != expected:
        raise ValueError(f"{dtype} payload size {len(blob)} != {expected}")
    return layout


def unpack_mx(dtype: str, blob: bytes | memoryview) -> MxTensor:
    layout = parse_mx_layout(dtype, blob)
    values = np.frombuffer(
        blob,
        dtype=np.uint8,
        count=layout.values_nbytes,
        offset=layout.values_offset,
    ).copy().reshape(layout.storage_shape)
    scales = np.frombuffer(
        blob,
        dtype=np.uint8,
        count=layout.scales_nbytes,
        offset=layout.scales_offset,
    ).copy().reshape(layout.scale_shape)
    return MxTensor(dtype, layout.shape, values, scales)


__all__ = [
    "MXFP4_DTYPE",
    "MXFP8_DTYPE",
    "MX_DTYPES",
    "MxTensor",
    "MxTensorLayout",
    "pack_mx",
    "mx_header_bytes",
    "parse_mx_layout",
    "unpack_mx",
    "validate_mx_shapes",
]
