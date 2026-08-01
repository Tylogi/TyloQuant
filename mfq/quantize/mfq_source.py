"""Streaming tensor sources for non-quantized (full-precision) MFQ files."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mfq.formats.assets import ASSET_DTYPE, MODEL_CONFIG_ASSET, is_asset_record
from mfq.formats.io import MMapTensorRecord, MMapTensorStore, open_mmap
from mfq.formats.mx import MX_DTYPES, MXFP4_DTYPE, parse_mx_layout
from mfq.quantize.mxfp import decode_mxfp4, decode_mxfp8

FULL_PRECISION_MFQ_DTYPES = frozenset(
    {"BF16", "F16", "F32", "I32", "I64", *MX_DTYPES}
)
_DENSE_DTYPES = {
    "BF16": np.dtype("<u2"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "I32": np.dtype("<i4"),
    "I64": np.dtype("<i8"),
}


@dataclass(frozen=True)
class FullPrecisionTensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source_index: int
    nbytes: int


def _dense_layout(
    store: MMapTensorStore,
    record: MMapTensorRecord,
) -> tuple[tuple[int, ...], int, np.dtype]:
    try:
        numpy_dtype = _DENSE_DTYPES[record.dtype]
    except KeyError as exc:
        raise ValueError(f"{record.name} is not a dense MFQ tensor") from exc
    mm = store.mmap_for(record)
    if record.nbytes < 4:
        raise ValueError(f"truncated dense MFQ tensor: {record.name}")
    ndim = struct.unpack_from("<I", mm, record.offset)[0]
    shape_bytes = 8 * ndim
    data_offset = record.offset + 4 + shape_bytes
    if ndim == 0 or data_offset > record.offset + record.nbytes:
        raise ValueError(f"invalid dense MFQ tensor rank: {record.name}")
    shape = tuple(
        int(value)
        for value in struct.unpack_from(f"<{ndim}q", mm, record.offset + 4)
    )
    if any(value <= 0 for value in shape):
        raise ValueError(f"invalid dense MFQ tensor shape: {record.name} {shape}")
    expected = 4 + shape_bytes + math.prod(shape) * numpy_dtype.itemsize
    if record.nbytes != expected:
        raise ValueError(
            f"dense MFQ tensor size mismatch for {record.name}: "
            f"{record.nbytes} != {expected}"
        )
    return shape, data_offset, numpy_dtype


class FullPrecisionMfqCheckpoint:
    """Validate and expose an MFQ containing no MFQ-quantized tensors."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.store = open_mmap(self.path)
        try:
            invalid = sorted(
                (name, record.dtype)
                for name, record in self.store.records.items()
                if not is_asset_record(name)
                and record.dtype not in FULL_PRECISION_MFQ_DTYPES
            )
            if invalid:
                preview = ", ".join(
                    f"{name}={dtype}" for name, dtype in invalid[:8]
                )
                raise ValueError(
                    "MFQ quantization input must contain only full-precision "
                    f"dtypes; found MFQ-quantized/unsupported records: {preview}"
                )
            self._info = {
                name: self._parse_info(record)
                for name, record in self.store.records.items()
                if not is_asset_record(name)
            }
        except Exception:
            self.store.close()
            raise

    def _parse_info(self, record: MMapTensorRecord) -> FullPrecisionTensorInfo:
        if record.dtype in MX_DTYPES:
            view = self.store.blob_view(record)
            try:
                shape = parse_mx_layout(record.dtype, view).shape
            finally:
                view.release()
        else:
            shape, _offset, _dtype = _dense_layout(self.store, record)
        return FullPrecisionTensorInfo(
            name=record.name,
            dtype=record.dtype,
            shape=shape,
            source_index=record.source_index,
            nbytes=record.nbytes,
        )

    @property
    def header(self):
        return self.store.header

    @property
    def infos(self) -> dict[str, FullPrecisionTensorInfo]:
        return dict(self._info)

    def info(self, name: str) -> FullPrecisionTensorInfo:
        try:
            return self._info[name]
        except KeyError as exc:
            raise KeyError(f"full-precision MFQ tensor is absent: {name}") from exc

    def tensor_source(self, name: str) -> FullPrecisionMfqTensorSource:
        return FullPrecisionMfqTensorSource(self, name)

    def model_config(self) -> dict[str, object]:
        record = self.store.records.get(MODEL_CONFIG_ASSET)
        if record is None or record.dtype != ASSET_DTYPE:
            value = self.header.extra.get("hf_config", {})
            return dict(value) if isinstance(value, dict) else {}
        value = json.loads(self.store.read_blob(MODEL_CONFIG_ASSET))
        if not isinstance(value, dict):
            raise ValueError("embedded MFQ model config is not a JSON object")
        return value

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> FullPrecisionMfqCheckpoint:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class FullPrecisionMfqTensorSource:
    """Row-readable dense, MXFP4, or block-scaled MXFP8 MFQ tensor."""

    def __init__(self, checkpoint: FullPrecisionMfqCheckpoint, name: str) -> None:
        self.checkpoint = checkpoint
        self.name = str(name)
        self.info = checkpoint.info(name)
        self.record = checkpoint.store.records[name]
        self.dtype_name = self.info.dtype
        self.shape = self.info.shape
        if not self.shape:
            raise ValueError(f"row source cannot expose a scalar: {name}")
        self.columns = int(self.shape[-1])
        self.rows = (
            int(math.prod(self.shape[:-1])) if len(self.shape) > 1 else 1
        )
        self._data_offset: int | None = None
        self._numpy_dtype: np.dtype | None = None
        self._mx_layout = None
        if self.dtype_name in MX_DTYPES:
            view = checkpoint.store.blob_view(self.record)
            try:
                self._mx_layout = parse_mx_layout(self.dtype_name, view)
            finally:
                view.release()
        else:
            shape, self._data_offset, self._numpy_dtype = _dense_layout(
                checkpoint.store, self.record
            )
            if shape != self.shape:
                raise AssertionError("dense MFQ shape changed while opening")

    def reshape(self, *shape: int) -> FullPrecisionMfqTensorSource:
        requested = tuple(int(value) for value in shape)
        if requested not in {self.shape, (self.rows, self.columns), (-1, self.columns)}:
            raise ValueError(f"unsupported full-precision MFQ reshape: {requested}")
        return self

    def _dense_rows(self, indices: np.ndarray) -> torch.Tensor:
        assert self._data_offset is not None and self._numpy_dtype is not None
        mm = self.checkpoint.store.mmap_for(self.record)
        mapped = np.frombuffer(
            mm,
            dtype=self._numpy_dtype,
            count=self.rows * self.columns,
            offset=self._data_offset,
        ).reshape(self.rows, self.columns)
        values = np.ascontiguousarray(mapped[indices])
        tensor = torch.from_numpy(values)
        if self.dtype_name == "BF16":
            tensor = tensor.view(torch.bfloat16)
        return tensor

    def _mx_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        layout = self._mx_layout
        assert layout is not None
        mm = self.checkpoint.store.mmap_for(self.record)
        base = self.record.offset
        values = np.frombuffer(
            mm,
            dtype=np.uint8,
            count=layout.values_nbytes,
            offset=base + layout.values_offset,
        ).reshape(layout.storage_shape)
        scales = np.frombuffer(
            mm,
            dtype=np.uint8,
            count=layout.scales_nbytes,
            offset=base + layout.scales_offset,
        ).reshape(layout.scale_shape)
        return values, scales

    def _read_contiguous_mx(
        self,
        start: int,
        stop: int,
        *,
        device: str | torch.device,
    ) -> torch.Tensor:
        values, scales = self._mx_arrays()
        if self.dtype_name == MXFP4_DTYPE:
            return decode_mxfp4(
                np.array(values[start:stop], copy=True, order="C"),
                np.array(scales[start:stop], copy=True, order="C"),
                device=device,
            )
        first_block = start // 128
        last_block = (stop - 1) // 128 if stop else first_block - 1
        return decode_mxfp8(
            np.array(values[start:stop], copy=True, order="C"),
            np.array(scales[first_block : last_block + 1], copy=True, order="C"),
            row_start=start,
            total_rows=self.rows,
            device=device,
        )

    def read_rows(
        self,
        start: int | np.ndarray,
        end: int | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        if end is not None:
            first = int(start)
            stop = int(end)
            if first < 0 or stop < first or stop > self.rows:
                raise IndexError(f"invalid MFQ row slice {first}:{stop} of {self.rows}")
            if self.dtype_name in MX_DTYPES:
                return self._read_contiguous_mx(first, stop, device=device)
            indices = np.arange(first, stop, dtype=np.int64)
            return self._dense_rows(indices).to(device=device)

        indices = np.asarray(start, dtype=np.int64).reshape(-1)
        if indices.size and (
            int(indices.min()) < 0 or int(indices.max()) >= self.rows
        ):
            raise IndexError(f"MFQ row indices fall outside [0, {self.rows})")
        if self.dtype_name not in MX_DTYPES:
            return self._dense_rows(indices).to(device=device)
        if not indices.size:
            return torch.empty((0, self.columns), device=device)
        # Tensor-codebook trainers use sampled row indices.  Decode consecutive
        # runs once and restore the caller's order without expanding the matrix.
        order = np.argsort(indices, kind="stable")
        sorted_indices = indices[order]
        sorted_chunks: list[torch.Tensor] = []
        run_start = 0
        while run_start < sorted_indices.size:
            run_end = run_start + 1
            while (
                run_end < sorted_indices.size
                and sorted_indices[run_end] == sorted_indices[run_end - 1] + 1
            ):
                run_end += 1
            first = int(sorted_indices[run_start])
            stop = int(sorted_indices[run_end - 1]) + 1
            sorted_chunks.append(
                self._read_contiguous_mx(first, stop, device=device)
            )
            run_start = run_end
        sorted_value = torch.cat(sorted_chunks, dim=0)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        return sorted_value[torch.as_tensor(inverse, device=device)]

    def read_expert_rows(
        self,
        expert: int,
        start: int,
        end: int,
        *,
        device: str | torch.device,
    ) -> torch.Tensor:
        if len(self.shape) != 3:
            raise ValueError(f"expert row reads require rank 3, got {self.shape}")
        n_experts, rows_per_expert, _columns = self.shape
        if (
            expert < 0
            or expert >= n_experts
            or start < 0
            or end < start
            or end > rows_per_expert
        ):
            raise IndexError(f"invalid expert row slice {expert}:{start}:{end}")
        return self.read_rows(
            expert * rows_per_expert + start,
            expert * rows_per_expert + end,
            device=device,
        )

    def tensor(self) -> torch.Tensor:
        value = self.read_rows(0, self.rows, device="cpu")
        return value.reshape(self.shape)

    def __getitem__(self, key: slice) -> torch.Tensor:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("full-precision MFQ source accepts contiguous slices")
        start = 0 if key.start is None else int(key.start)
        stop = self.rows if key.stop is None else int(key.stop)
        return self.read_rows(start, stop, device="cpu")


__all__ = [
    "FULL_PRECISION_MFQ_DTYPES",
    "FullPrecisionMfqCheckpoint",
    "FullPrecisionMfqTensorSource",
    "FullPrecisionTensorInfo",
]
