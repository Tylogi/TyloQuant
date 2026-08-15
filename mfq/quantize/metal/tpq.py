"""Metal nearest-codeword assignment for offline TPQ quantization."""

from __future__ import annotations

from functools import lru_cache

import torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

inline bool better(float error, uint index, float best_error, uint best_index) {
  return error < best_error || (error == best_error && index < best_index);
}

kernel void tpq_assign(
    device const float* points,
    device const float* codebook,
    device int* out_labels,
    device float* out_errors,
    constant uint& vector_size,
    constant uint& entries,
    uint tid [[thread_index_in_threadgroup]],
    uint point [[threadgroup_position_in_grid]]) {
  threadgroup float error_parts[256];
  threadgroup uint index_parts[256];
  const uint point_offset = point * vector_size;
  float best_error = INFINITY;
  uint best_index = 0;
  for (uint entry = tid; entry < entries; entry += 256) {
    const uint code_offset = entry * vector_size;
    float error = 0.0f;
    for (uint coordinate = 0; coordinate < vector_size; ++coordinate) {
      const float residual =
          points[point_offset + coordinate] - codebook[code_offset + coordinate];
      error = fma(residual, residual, error);
    }
    if (better(error, entry, best_error, best_index)) {
      best_error = error;
      best_index = entry;
    }
  }
  error_parts[tid] = best_error;
  index_parts[tid] = best_index;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride && better(
            error_parts[tid + stride],
            index_parts[tid + stride],
            error_parts[tid],
            index_parts[tid])) {
      error_parts[tid] = error_parts[tid + stride];
      index_parts[tid] = index_parts[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    out_labels[point] = int(index_parts[0]);
    out_errors[point] = error_parts[0];
  }
}

kernel void tpq_int4(
    device const float* weight,
    device uchar* out_packed,
    device half* out_scales,
    constant uint& columns,
    constant uint& groups_per_row,
    constant uint& group_size,
    uint tid [[thread_index_in_threadgroup]],
    uint flat_group [[threadgroup_position_in_grid]]) {
  threadgroup float maximum_parts[1024];
  const uint row = flat_group / groups_per_row;
  const uint group = flat_group - row * groups_per_row;
  const uint group_offset = row * columns + group * group_size;
  maximum_parts[tid] = abs(weight[group_offset + tid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = group_size >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      maximum_parts[tid] = max(
          maximum_parts[tid], maximum_parts[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float scale = max(maximum_parts[0] / 7.0f, 1.0e-12f);
  if (tid == 0) {
    out_scales[flat_group] = half(scale);
  }
  if (tid < group_size / 2) {
    const float first = weight[group_offset + tid * 2] / scale;
    const float second = weight[group_offset + tid * 2 + 1] / scale;
    const int q0 = clamp(int(rint(first)), -7, 7) + 8;
    const int q1 = clamp(int(rint(second)), -7, 7) + 8;
    const uint packed_offset =
        row * (columns / 2) + group * (group_size / 2) + tid;
    out_packed[packed_offset] = uchar(q0 | (q1 << 4));
  }
}
"""


@lru_cache(maxsize=1)
def _library():
    if not torch.backends.mps.is_available():
        raise RuntimeError("TPQ Metal assignment requires MPS")
    return torch.mps.compile_shader(_SOURCE)


def assign(
    points: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nearest codeword labels and squared errors on MPS."""

    if points.device.type != "mps" or points.dtype != torch.float32:
        raise TypeError("TPQ Metal assignment expects MPS float32 points")
    if codebook.device != points.device or codebook.dtype != torch.float32:
        raise TypeError("TPQ Metal assignment expects an MPS float32 codebook")
    if points.ndim != 2 or codebook.ndim != 2:
        raise ValueError("TPQ Metal assignment expects two matrices")
    count, vector_size = map(int, points.shape)
    entries, code_width = map(int, codebook.shape)
    if count <= 0 or entries <= 1 or vector_size <= 0 or code_width != vector_size:
        raise ValueError("invalid TPQ Metal assignment geometry")
    labels_i32 = torch.empty(count, device="mps", dtype=torch.int32)
    errors = torch.empty(count, device="mps", dtype=torch.float32)
    _library().tpq_assign(
        points.contiguous(),
        codebook.contiguous(),
        labels_i32,
        errors,
        vector_size,
        entries,
        threads=count * 256,
        group_size=256,
    )
    return labels_i32.to(torch.int64), errors


def quantize_int4(
    weight: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a row-major matrix with TPQ's symmetric int4 rule."""

    if weight.device.type != "mps" or weight.dtype != torch.float32:
        raise TypeError("TPQ-I4 Metal quantization expects an MPS float32 matrix")
    if weight.ndim != 2:
        raise ValueError("TPQ-I4 Metal quantization expects a matrix")
    rows, columns = map(int, weight.shape)
    if (
        group_size <= 0
        or group_size > 1024
        or group_size & (group_size - 1)
        or group_size % 2
        or columns % group_size
    ):
        raise ValueError(
            "TPQ-I4 Metal group size must be an even power of two up to 1024"
        )
    groups_per_row = columns // group_size
    groups = rows * groups_per_row
    packed = torch.empty((rows, columns // 2), device="mps", dtype=torch.uint8)
    scales = torch.empty(
        (rows, groups_per_row), device="mps", dtype=torch.float16
    )
    _library().tpq_int4(
        weight.contiguous(),
        packed,
        scales,
        columns,
        groups_per_row,
        group_size,
        threads=groups * group_size,
        group_size=group_size,
    )
    return packed, scales


__all__ = ["assign", "quantize_int4"]
