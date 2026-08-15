"""Fused Metal search kernels used by NINT quantization."""

from __future__ import annotations

from functools import lru_cache

import torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

inline float quant_level(float value, float minimum, float iscale, uint nmax) {
  return clamp(rint(iscale * (value - minimum)), 0.0f, float(nmax));
}

inline float positive_level(float value, float iscale, uint nmax) {
  return clamp(rint(iscale * value), 0.0f, float(nmax));
}

kernel void nint_make_qkx(
    device const float* x,
    device const float* weight,
    device float* out_scale,
    device float* out_minimum,
    constant uint& groups,
    constant uint& group_size,
    constant uint& nmax,
    constant float& rmin,
    constant float& rdelta,
    constant uint& nstep,
    constant uint& strict_range,
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  if (group >= groups) return;
  const ulong base = ulong(group) * ulong(group_size);
  const uint second = lane + 32;
  const bool valid_first = lane < group_size;
  const bool valid_second = second < group_size;
  const float first_value = valid_first ? x[base + lane] : 0.0f;
  const float second_value = valid_second ? x[base + second] : 0.0f;
  const float first_weight = valid_first ? weight[base + lane] : 0.0f;
  const float second_weight = valid_second ? weight[base + second] : 0.0f;

  float local_minimum = valid_first ? first_value : INFINITY;
  float local_maximum = valid_first ? first_value : -INFINITY;
  if (valid_second) {
    local_minimum = min(local_minimum, second_value);
    local_maximum = max(local_maximum, second_value);
  }
  float minimum = simd_min(local_minimum);
  const float maximum = simd_max(local_maximum);
  const float sum_weight = simd_sum(first_weight + second_weight);
  const float sum_x = simd_sum(
      first_weight * first_value + second_weight * second_value);
  minimum = min(minimum, 0.0f);
  const bool degenerate =
      (strict_range != 0u ? maximum <= minimum : maximum == minimum)
      || sum_weight <= 0.0f;
  const float range = degenerate ? 1.0f : maximum - minimum;
  const float reciprocal_range = 1.0f / range;
  const float initial_iscale = float(nmax) * reciprocal_range;
  const float initial_scale = 1.0f / initial_iscale;
  const float first_initial_level = valid_first
      ? quant_level(first_value, minimum, initial_iscale, nmax) : 0.0f;
  const float second_initial_level = valid_second
      ? quant_level(second_value, minimum, initial_iscale, nmax) : 0.0f;
  const float first_initial_diff =
      initial_scale * first_initial_level + minimum - first_value;
  const float second_initial_diff =
      initial_scale * second_initial_level + minimum - second_value;
  float best_error = simd_sum(
      first_weight * first_initial_diff * first_initial_diff
      + second_weight * second_initial_diff * second_initial_diff);
  float best_scale = initial_scale;
  float best_minimum = minimum;

  for (uint step = 0; step <= nstep; ++step) {
    const float numerator = rmin + rdelta * float(step) + float(nmax);
    const float iscale = numerator * reciprocal_range;
    const float first_level = valid_first
        ? quant_level(first_value, minimum, iscale, nmax) : 0.0f;
    const float second_level = valid_second
        ? quant_level(second_value, minimum, iscale, nmax) : 0.0f;
    const float first_weighted_level = first_weight * first_level;
    const float second_weighted_level = second_weight * second_level;
    const float sum_l = simd_sum(first_weighted_level + second_weighted_level);
    const float sum_l2 = simd_sum(
        first_weighted_level * first_level
        + second_weighted_level * second_level);
    const float sum_xl = simd_sum(
        first_weighted_level * first_value
        + second_weighted_level * second_value);
    float candidate_scale = 0.0f;
    float candidate_minimum = 0.0f;
    uint candidate_valid = 0;
    if (lane == 0) {
      const float determinant = sum_weight * sum_l2 - sum_l * sum_l;
      candidate_valid = determinant > 0.0f;
      if (candidate_valid) {
        candidate_scale = (sum_weight * sum_xl - sum_x * sum_l) / determinant;
        candidate_minimum = (sum_l2 * sum_x - sum_l * sum_xl) / determinant;
        if (candidate_minimum > 0.0f) {
          candidate_scale = sum_l2 > 0.0f ? sum_xl / sum_l2 : 0.0f;
          candidate_minimum = 0.0f;
        }
      }
    }
    candidate_scale = simd_broadcast(candidate_scale, 0);
    candidate_minimum = simd_broadcast(candidate_minimum, 0);
    candidate_valid = simd_broadcast(candidate_valid, 0);
    const float first_diff =
        candidate_scale * first_level + candidate_minimum - first_value;
    const float second_diff =
        candidate_scale * second_level + candidate_minimum - second_value;
    const float candidate_error = simd_sum(
        first_weight * first_diff * first_diff
        + second_weight * second_diff * second_diff);
    if (lane == 0 && candidate_valid && candidate_error < best_error) {
      best_error = candidate_error;
      best_scale = candidate_scale;
      best_minimum = candidate_minimum;
    }
  }
  if (lane == 0) {
    out_scale[group] = degenerate ? 0.0f : best_scale;
    out_minimum[group] = degenerate ? min(minimum, 0.0f) : best_minimum;
  }
}

kernel void nint_make_qp(
    device const float* x,
    device const float* weight,
    device float* out_scale,
    device int* levels,
    constant uint& rows,
    constant uint& width,
    constant uint& nmax,
    uint lane [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  if (row >= rows) return;
  const ulong base = ulong(row) * ulong(width);
  float local_maximum = -INFINITY;
  for (uint index = lane; index < width; index += 32) {
    local_maximum = max(local_maximum, x[base + index]);
  }
  const float maximum = simd_max(local_maximum);
  const bool active = maximum >= 1.0e-15f;
  const float safe_maximum = active ? maximum : 1.0f;
  const float reciprocal_maximum = 1.0f / safe_maximum;
  float iscale = float(nmax) * reciprocal_maximum;
  float scale = 1.0f / iscale;
  float local_error = 0.0f;
  for (uint index = lane; index < width; index += 32) {
    const float value = x[base + index];
    const float level = positive_level(value, iscale, nmax);
    const float difference = value - scale * level;
    local_error += weight[base + index] * difference * difference;
  }
  float best_error = simd_sum(local_error);
  for (int offset = -4; offset <= 4; ++offset) {
    if (offset == 0) continue;
    const float candidate_iscale =
        (float(nmax) + 0.1f * float(offset)) * reciprocal_maximum;
    const float candidate_scale = 1.0f / candidate_iscale;
    float candidate_local_error = 0.0f;
    for (uint index = lane; index < width; index += 32) {
      const float value = x[base + index];
      const float level = positive_level(value, candidate_iscale, nmax);
      const float difference = value - candidate_scale * level;
      candidate_local_error += weight[base + index] * difference * difference;
    }
    const float candidate_error = simd_sum(candidate_local_error);
    if (lane == 0 && active && candidate_error < best_error) {
      best_error = candidate_error;
      iscale = candidate_iscale;
    }
    iscale = simd_broadcast(iscale, 0);
  }

  float local_lx = 0.0f;
  float local_l2 = 0.0f;
  for (uint index = lane; index < width; index += 32) {
    const ulong position = base + index;
    const float value = x[position];
    const int level = active ? int(positive_level(value, iscale, nmax)) : 0;
    levels[position] = level;
    const float level_f = float(level);
    local_lx += weight[position] * value * level_f;
    local_l2 += weight[position] * level_f * level_f;
  }
  float sum_lx = simd_sum(local_lx);
  float sum_l2 = simd_sum(local_l2);
  threadgroup_barrier(mem_flags::mem_device);
  if (lane == 0) {
    for (uint pass = 0; pass < 5; ++pass) {
      for (uint index = 0; index < width; ++index) {
        const ulong position = base + index;
        const int old_level = levels[position];
        const float old_level_f = float(old_level);
        const float objective = weight[position];
        const float value = x[position];
        const float candidate_lx = sum_lx - objective * value * old_level_f;
        const float candidate_l2 = sum_l2 - objective * old_level_f * old_level_f;
        if (candidate_lx <= 0.0f || candidate_l2 <= 0.0f) continue;
        const int new_level = int(positive_level(
            value, candidate_l2 / candidate_lx, nmax));
        if (new_level == old_level) continue;
        const float new_level_f = float(new_level);
        const float updated_lx = candidate_lx + objective * value * new_level_f;
        const float updated_l2 = candidate_l2 + objective * new_level_f * new_level_f;
        if (updated_lx * updated_lx * sum_l2 > sum_lx * sum_lx * updated_l2) {
          levels[position] = new_level;
          sum_lx = updated_lx;
          sum_l2 = updated_l2;
        }
      }
    }
    out_scale[row] = active && sum_l2 > 0.0f ? sum_lx / sum_l2 : 0.0f;
  }
}
"""


@lru_cache(maxsize=1)
def _library():
    if not torch.backends.mps.is_available():
        raise RuntimeError("NINT Metal quantization requires MPS")
    return torch.mps.compile_shader(_SOURCE)


def _make_qkx(
    value: torch.Tensor,
    weight: torch.Tensor,
    nmax: int,
    rmin: float,
    rdelta: float,
    nstep: int,
    *,
    strict_range: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.device.type != "mps" or value.dtype != torch.float32 or value.ndim != 3:
        raise TypeError("NINT qkx Metal expects MPS float32 [rows, groups, width]")
    if weight.shape != value.shape or weight.dtype != torch.float32:
        raise TypeError("NINT qkx Metal objective weights must match the values")
    rows, groups_per_row, width = map(int, value.shape)
    if not 0 < width <= 64:
        raise ValueError("NINT qkx Metal group width must be in [1,64]")
    groups = rows * groups_per_row
    scale = torch.empty((rows, groups_per_row), device="mps", dtype=torch.float32)
    minimum = torch.empty_like(scale)
    _library().nint_make_qkx(
        value.contiguous(),
        weight.contiguous(),
        scale,
        minimum,
        groups,
        width,
        int(nmax),
        float(rmin),
        float(rdelta),
        int(nstep),
        int(strict_range),
        threads=groups * 32,
        group_size=32,
    )
    return scale, minimum


def make_qkx2(
    value: torch.Tensor,
    weight: torch.Tensor,
    nmax: int = 15,
    rmin: float | None = None,
    rdelta: float = 0.1,
    nstep: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the ordinary weighted NINT group search in one Metal dispatch."""

    if rmin is None or nstep is None:
        default_rmin, default_nstep = (
            (-1.0, 20) if int(nmax) <= 15 else (-0.5, 15)
        )
        if rmin is None:
            rmin = default_rmin
        if nstep is None:
            nstep = default_nstep
    return _make_qkx(
        value,
        weight,
        nmax,
        float(rmin),
        float(rdelta),
        int(nstep),
        strict_range=False,
    )


def make_qkx3(
    value: torch.Tensor,
    weight: torch.Tensor,
    nmax: int,
    rmin: float = -0.9,
    rdelta: float = 0.05,
    nstep: int = 36,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the imatrix-weighted NINT group search in one Metal dispatch."""

    return _make_qkx(
        value,
        weight,
        nmax,
        rmin,
        rdelta,
        nstep,
        strict_range=True,
    )


def make_qp(
    value: torch.Tensor,
    weight: torch.Tensor,
    nmax: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.device.type != "mps" or value.dtype != torch.float32 or value.ndim != 2:
        raise TypeError("NINT qp Metal expects MPS float32 [rows, width]")
    if weight.shape != value.shape or weight.dtype != torch.float32:
        raise TypeError("NINT qp Metal objective weights must match the values")
    rows, width = map(int, value.shape)
    scale = torch.empty((rows,), device="mps", dtype=torch.float32)
    levels = torch.empty(value.shape, device="mps", dtype=torch.int32)
    _library().nint_make_qp(
        value.contiguous(),
        weight.contiguous(),
        scale,
        levels,
        rows,
        width,
        int(nmax),
        threads=rows * 32,
        group_size=32,
    )
    return scale, levels


__all__ = ["make_qkx2", "make_qkx3", "make_qp"]
