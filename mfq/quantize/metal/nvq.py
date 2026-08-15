"""Metal search and reassignment kernels for offline NVQ quantization."""

from __future__ import annotations

from functools import lru_cache

import torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

inline bool better(float error, uint index, float best_error, uint best_index) {
  return error < best_error || (error == best_error && index < best_index);
}

inline void nvq_assign_vectors(
    device const float* group_value,
    device const float* group_weight,
    device const char* codebook,
    float scale,
    uint valid_group,
    uint vector_size,
    uint vectors_per_group,
    uint entries,
    threadgroup uint* selected,
    uint tid) {
  const uint vector = tid >> 5;
  const uint lane = tid & 31;
  if (vector < vectors_per_group) {
    const int valid = max(
        0, min(int(vector_size), int(valid_group) - int(vector * vector_size)));
    float best_error = INFINITY;
    uint best_index = 0;
    for (uint entry = lane; entry < entries; entry += 32) {
      float dot = 0.0f;
      float norm = 0.0f;
      for (int coordinate = 0; coordinate < valid; ++coordinate) {
        const float code = float(codebook[
            entry * vector_size + uint(coordinate)]);
        const uint position = vector * vector_size + uint(coordinate);
        const float source = group_value[position];
        const float objective = group_weight[position];
        dot = fma(objective * source, code, dot);
        norm = fma(objective * code, code, norm);
      }
      const float error = fma(scale * scale, norm, -2.0f * scale * dot);
      if (better(error, entry, best_error, best_index)) {
        best_error = error;
        best_index = entry;
      }
    }
    for (uint offset = 16; offset > 0; offset >>= 1) {
      const float other_error = simd_shuffle_down(best_error, offset);
      const uint other_index = simd_shuffle_down(best_index, offset);
      if (lane < offset && better(
              other_error, other_index, best_error, best_index)) {
        best_error = other_error;
        best_index = other_index;
      }
    }
    if (lane == 0) {
      selected[vector] = best_index;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void nvq_search(
    device const float* xgroup,
    device const float* wgroup,
    device const char* codebook,
    device float* out_scale,
    device int* out_indices,
    constant uint& groups_per_row,
    constant uint& valid_last,
    constant uint& vector_size,
    constant uint& vectors_per_group,
    constant uint& entries,
    constant uint& search_steps,
    constant float& qmax,
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup uint selected[6];
  threadgroup uint best_indices[6];
  threadgroup float current_scale;
  threadgroup float best_scale;
  threadgroup float best_error;
  const uint valid_group = group % groups_per_row == groups_per_row - 1
      ? valid_last : 24;
  device const float* group_value = xgroup + group * 24;
  device const float* group_weight = wgroup + group * 24;
  if (tid == 0) {
    best_error = INFINITY;
    best_scale = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint step = 0; step < search_steps; ++step) {
    if (tid == 0) {
      float max_abs = 0.0f;
      for (uint position = 0; position < valid_group; ++position) {
        max_abs = max(max_abs, abs(group_value[position]));
      }
      const float offset = search_steps == 1
          ? -0.12f * qmax
          : -0.12f * qmax
              + float(step) * (0.24f * qmax / float(search_steps - 1));
      current_scale = max_abs > 0.0f ? max_abs / (qmax + offset) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    nvq_assign_vectors(
        group_value, group_weight, codebook, current_scale, valid_group,
        vector_size, vectors_per_group, entries, selected, tid);
    if (tid == 0) {
      float numerator = 0.0f;
      float denominator = 0.0f;
      for (uint position = 0; position < valid_group; ++position) {
        const uint vector = position / vector_size;
        const uint coordinate = position - vector * vector_size;
        const float code = float(codebook[
            selected[vector] * vector_size + coordinate]);
        const float objective = group_weight[position];
        numerator = fma(objective * group_value[position], code, numerator);
        denominator = fma(objective * code, code, denominator);
      }
      current_scale = denominator > 0.0f
          ? max(numerator / denominator, 0.0f) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    nvq_assign_vectors(
        group_value, group_weight, codebook, current_scale, valid_group,
        vector_size, vectors_per_group, entries, selected, tid);
    if (tid == 0) {
      float numerator = 0.0f;
      float denominator = 0.0f;
      for (uint position = 0; position < valid_group; ++position) {
        const uint vector = position / vector_size;
        const uint coordinate = position - vector * vector_size;
        const float code = float(codebook[
            selected[vector] * vector_size + coordinate]);
        const float objective = group_weight[position];
        numerator = fma(objective * group_value[position], code, numerator);
        denominator = fma(objective * code, code, denominator);
      }
      current_scale = denominator > 0.0f
          ? max(numerator / denominator, 0.0f) : 0.0f;
      float error = 0.0f;
      for (uint position = 0; position < valid_group; ++position) {
        const uint vector = position / vector_size;
        const uint coordinate = position - vector * vector_size;
        const float code = float(codebook[
            selected[vector] * vector_size + coordinate]);
        const float residual = current_scale * code - group_value[position];
        error = fma(group_weight[position] * residual, residual, error);
      }
      if (error < best_error) {
        best_error = error;
        best_scale = current_scale;
        for (uint vector = 0; vector < vectors_per_group; ++vector) {
          best_indices[vector] = selected[vector];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    out_scale[group] = best_scale;
    for (uint vector = 0; vector < vectors_per_group; ++vector) {
      out_indices[group * vectors_per_group + vector] =
          int(best_indices[vector]);
    }
  }
}

kernel void nvq_reassign(
    device const float* xgroup,
    device const float* wgroup,
    device const float* scale,
    device const char* codebook,
    device int* out_indices,
    constant uint& groups_per_row,
    constant uint& valid_last,
    constant uint& vector_size,
    constant uint& vectors_per_group,
    constant uint& entries,
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup uint selected[6];
  const uint valid_group = group % groups_per_row == groups_per_row - 1
      ? valid_last : 24;
  nvq_assign_vectors(
      xgroup + group * 24, wgroup + group * 24, codebook, scale[group],
      valid_group, vector_size, vectors_per_group, entries, selected, tid);
  if (tid == 0) {
    for (uint vector = 0; vector < vectors_per_group; ++vector) {
      out_indices[group * vectors_per_group + vector] = int(selected[vector]);
    }
  }
}

kernel void nvq1_l_assign(
    device const float* xgroup,
    device const float* wgroup,
    device const float* group_anchor,
    device const char* codebook,
    device uchar* out_scale,
    device uchar* out_delta,
    device int* out_indices,
    constant uint& groups_per_row,
    constant uint& valid_last,
    constant uint& q_count,
    constant float& delta_value,
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup float vector_error[2 * 3 * 16];
  threadgroup uint vector_index[2 * 3 * 16];
  const uint vector = tid >> 5;
  const uint lane = tid & 31;
  const uint valid_group = group % groups_per_row == groups_per_row - 1
      ? valid_last : 24;
  const int valid = max(0, min(8, int(valid_group) - int(vector * 8)));
  device const float* group_value = xgroup + group * 24;
  device const float* group_weight = wgroup + group * 24;
  const float anchor = group_anchor[group];
  for (uint delta_bit = 0; delta_bit < 2; ++delta_bit) {
    const float delta = delta_bit == 0 ? delta_value : -delta_value;
    for (uint q = 0; q < q_count; ++q) {
      float best_error = INFINITY;
      uint best_index = 0;
      for (uint entry = lane; entry < 2048; entry += 32) {
        float dot = 0.0f;
        float norm = 0.0f;
        for (int coordinate = 0; coordinate < valid; ++coordinate) {
          const uint position = vector * 8 + uint(coordinate);
          const float code = float(codebook[entry * 8 + uint(coordinate)])
                           + delta;
          const float source = group_value[position];
          const float objective = group_weight[position];
          dot = fma(objective * source, code, dot);
          norm = fma(objective * code, code, norm);
        }
        const float scale = anchor * float(q);
        const float error = fma(scale * scale, norm, -2.0f * scale * dot);
        if (better(error, entry, best_error, best_index)) {
          best_error = error;
          best_index = entry;
        }
      }
      for (uint offset = 16; offset > 0; offset >>= 1) {
        const float other_error = simd_shuffle_down(best_error, offset);
        const uint other_index = simd_shuffle_down(best_index, offset);
        if (lane < offset && better(
                other_error, other_index, best_error, best_index)) {
          best_error = other_error;
          best_index = other_index;
        }
      }
      if (lane == 0) {
        const uint offset = (delta_bit * 3 + vector) * 16 + q;
        vector_error[offset] = best_error;
        vector_index[offset] = best_index;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float best_error = INFINITY;
    uint best_delta = 0;
    uint best_q = 0;
    for (uint delta_bit = 0; delta_bit < 2; ++delta_bit) {
      for (uint q = 0; q < q_count; ++q) {
        float error = 0.0f;
        for (uint vector_id = 0; vector_id < 3; ++vector_id) {
          error += vector_error[(delta_bit * 3 + vector_id) * 16 + q];
        }
        if (error < best_error) {
          best_error = error;
          best_delta = delta_bit;
          best_q = q;
        }
      }
    }
    out_scale[group] = uchar(best_q);
    out_delta[group] = uchar(best_delta);
    for (uint vector_id = 0; vector_id < 3; ++vector_id) {
      out_indices[group * 3 + vector_id] = int(
          vector_index[(best_delta * 3 + vector_id) * 16 + best_q]);
    }
  }
}

kernel void nvq1_s_assign(
    device const float* xgroup,
    device const float* wgroup,
    device const float* group_anchor,
    device const char* codebooks,
    device uchar* out_scale,
    device uchar* out_delta,
    device int* out_indices,
    constant uint& groups_per_row,
    constant uint& valid_last,
    constant float& delta_value,
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup float vector_error[2 * 3 * 16];
  threadgroup uint vector_index[2 * 3 * 16];
  const uint vector = tid >> 5;
  const uint lane = tid & 31;
  const uint valid_group = group % groups_per_row == groups_per_row - 1
      ? valid_last : 24;
  const int valid = max(0, min(8, int(valid_group) - int(vector * 8)));
  device const float* group_value = xgroup + group * 24;
  device const float* group_weight = wgroup + group * 24;
  const float anchor = group_anchor[group];
  for (uint delta_bit = 0; delta_bit < 2; ++delta_bit) {
    const float delta = delta_bit == 0 ? delta_value : -delta_value;
    for (uint q = 0; q < 16; ++q) {
      float best_error = INFINITY;
      uint best_index = 0;
      for (uint entry = lane; entry < 512; entry += 32) {
        float dot = 0.0f;
        float norm = 0.0f;
        const uint code_offset = (delta_bit * 512 + entry) * 8;
        for (int coordinate = 0; coordinate < valid; ++coordinate) {
          const uint position = vector * 8 + uint(coordinate);
          const float code = float(codebooks[
              code_offset + uint(coordinate)]) + delta;
          const float source = group_value[position];
          const float objective = group_weight[position];
          dot = fma(objective * source, code, dot);
          norm = fma(objective * code, code, norm);
        }
        const float scale = anchor * float(q);
        const float error = fma(scale * scale, norm, -2.0f * scale * dot);
        if (better(error, entry, best_error, best_index)) {
          best_error = error;
          best_index = entry;
        }
      }
      for (uint offset = 16; offset > 0; offset >>= 1) {
        const float other_error = simd_shuffle_down(best_error, offset);
        const uint other_index = simd_shuffle_down(best_index, offset);
        if (lane < offset && better(
                other_error, other_index, best_error, best_index)) {
          best_error = other_error;
          best_index = other_index;
        }
      }
      if (lane == 0) {
        const uint offset = (delta_bit * 3 + vector) * 16 + q;
        vector_error[offset] = best_error;
        vector_index[offset] = best_index;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float best_error = INFINITY;
    uint best_delta = 0;
    uint best_q = 0;
    for (uint delta_bit = 0; delta_bit < 2; ++delta_bit) {
      for (uint q = 0; q < 16; ++q) {
        float error = 0.0f;
        for (uint vector_id = 0; vector_id < 3; ++vector_id) {
          error += vector_error[(delta_bit * 3 + vector_id) * 16 + q];
        }
        if (error < best_error) {
          best_error = error;
          best_delta = delta_bit;
          best_q = q;
        }
      }
    }
    out_scale[group] = uchar(best_q);
    out_delta[group] = uchar(best_delta);
    for (uint vector_id = 0; vector_id < 3; ++vector_id) {
      out_indices[group * 3 + vector_id] = int(
          vector_index[(best_delta * 3 + vector_id) * 16 + best_q]);
    }
  }
}
"""


@lru_cache(maxsize=1)
def _library():
    if not torch.backends.mps.is_available():
        raise RuntimeError("NVQ Metal quantization requires MPS")
    return torch.mps.compile_shader(_SOURCE)


def _validate_common(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    codebook: torch.Tensor,
    groups_per_row: int,
    valid_last: int,
) -> tuple[int, int, int]:
    if xgroup.device.type != "mps" or xgroup.dtype != torch.float32:
        raise TypeError("NVQ Metal kernels expect MPS float32 groups")
    if tuple(xgroup.shape) != tuple(wgroup.shape) or xgroup.shape[1] != 24:
        raise ValueError("NVQ Metal groups must have matching [groups,24] shapes")
    entries, vector_size = map(int, codebook.shape)
    if vector_size not in {4, 8} or 24 % vector_size:
        raise ValueError("NVQ Metal vector size must be 4 or 8")
    if groups_per_row <= 0 or not 0 < valid_last <= 24:
        raise ValueError("invalid NVQ Metal row geometry")
    return entries, vector_size, 24 // vector_size


def nvq_search(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    codebook: torch.Tensor,
    groups_per_row: int,
    valid_last: int,
    search_steps: int,
    qmax: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Search initial NVQ group scales and vector indices."""

    entries, vector_size, vectors_per_group = _validate_common(
        xgroup, wgroup, codebook, groups_per_row, valid_last
    )
    if search_steps <= 0 or qmax <= 0:
        raise ValueError("invalid NVQ Metal scale search configuration")
    groups = int(xgroup.shape[0])
    scale = torch.empty(groups, device="mps", dtype=torch.float32)
    indices_i32 = torch.empty(
        (groups, vectors_per_group), device="mps", dtype=torch.int32
    )
    group_size = vectors_per_group * 32
    _library().nvq_search(
        xgroup.contiguous(),
        wgroup.contiguous(),
        codebook.contiguous(),
        scale,
        indices_i32,
        int(groups_per_row),
        int(valid_last),
        vector_size,
        vectors_per_group,
        entries,
        int(search_steps),
        float(qmax),
        threads=groups * group_size,
        group_size=group_size,
    )
    return scale, indices_i32.to(torch.int64)


def nvq_reassign(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    scale: torch.Tensor,
    codebook: torch.Tensor,
    groups_per_row: int,
    valid_last: int,
) -> torch.Tensor:
    """Reassign NVQ vectors at fixed group scales."""

    entries, vector_size, vectors_per_group = _validate_common(
        xgroup, wgroup, codebook, groups_per_row, valid_last
    )
    groups = int(xgroup.shape[0])
    indices_i32 = torch.empty(
        (groups, vectors_per_group), device="mps", dtype=torch.int32
    )
    group_size = vectors_per_group * 32
    _library().nvq_reassign(
        xgroup.contiguous(),
        wgroup.contiguous(),
        scale.contiguous(),
        codebook.contiguous(),
        indices_i32,
        int(groups_per_row),
        int(valid_last),
        vector_size,
        vectors_per_group,
        entries,
        threads=groups * group_size,
        group_size=group_size,
    )
    return indices_i32.to(torch.int64)


def nvq1_l_assign(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    group_anchor: torch.Tensor,
    codebook: torch.Tensor,
    groups_per_row: int,
    valid_last: int,
    sub_bits: int,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Jointly assign NVQ1-L scale states, delta signs, and code indices."""

    if tuple(codebook.shape) != (2048, 8):
        raise ValueError("NVQ1-L Metal codebook must have shape [2048,8]")
    _validate_common(xgroup, wgroup, codebook, groups_per_row, valid_last)
    if sub_bits not in {3, 4}:
        raise ValueError("NVQ1-L Metal sub_bits must be 3 or 4")
    groups = int(xgroup.shape[0])
    scale = torch.empty(groups, device="mps", dtype=torch.uint8)
    delta_sign = torch.empty_like(scale)
    indices_i32 = torch.empty((groups, 3), device="mps", dtype=torch.int32)
    _library().nvq1_l_assign(
        xgroup.contiguous(),
        wgroup.contiguous(),
        group_anchor.contiguous(),
        codebook.contiguous(),
        scale,
        delta_sign,
        indices_i32,
        int(groups_per_row),
        int(valid_last),
        1 << int(sub_bits),
        float(delta),
        threads=groups * 96,
        group_size=96,
    )
    return scale, delta_sign, indices_i32.to(torch.int64)


def nvq1_s_assign(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    group_anchor: torch.Tensor,
    codebooks: torch.Tensor,
    groups_per_row: int,
    valid_last: int,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign NVQ1-S scale states, codebook banks, and code indices."""

    if tuple(codebooks.shape) != (2, 512, 8):
        raise ValueError("NVQ1-S Metal codebooks must have shape [2,512,8]")
    _validate_common(
        xgroup,
        wgroup,
        codebooks[0],
        groups_per_row,
        valid_last,
    )
    groups = int(xgroup.shape[0])
    scale = torch.empty(groups, device="mps", dtype=torch.uint8)
    delta_sign = torch.empty_like(scale)
    indices_i32 = torch.empty((groups, 3), device="mps", dtype=torch.int32)
    _library().nvq1_s_assign(
        xgroup.contiguous(),
        wgroup.contiguous(),
        group_anchor.contiguous(),
        codebooks.contiguous(),
        scale,
        delta_sign,
        indices_i32,
        int(groups_per_row),
        int(valid_last),
        float(delta),
        threads=groups * 96,
        group_size=96,
    )
    return scale, delta_sign, indices_i32.to(torch.int64)


__all__ = [
    "nvq1_l_assign",
    "nvq1_s_assign",
    "nvq_reassign",
    "nvq_search",
]
