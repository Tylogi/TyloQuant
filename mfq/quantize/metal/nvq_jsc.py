"""Fused Metal assignment kernels for fixed NVQ-JSC tables."""

from __future__ import annotations

from functools import lru_cache

import torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

template <typename IndexT>
inline void nvq2j_assign_impl(
    device const float* value,
    device const float* weight,
    device const float* anchor,
    device const float* scale_lut,
    device const uchar* bank_for_state,
    device const char* codebooks,
    device uchar* out_state,
    device IndexT* out_indices,
    threadgroup float* vector_error,
    threadgroup uint* vector_index,
    constant uint& padded_width,
    constant uint& valid_width,
    constant uint& groups_per_row,
    constant uint& entries,
    uint tid,
    uint flat_group) {
  const uint row = flat_group / groups_per_row;
  const uint group = flat_group - row * groups_per_row;
  const uint vector = tid >> 5;
  const uint lane = tid & 31;
  const uint first = group * 24;
  const uint valid_group = min(uint(24), valid_width - first);
  const int valid_vector = max(0, min(8, int(valid_group) - int(vector * 8)));

  for (uint bank = 0; bank < 4; ++bank) {
    uint states[4];
    uint state_count = 0;
    for (uint state = 0; state < 16; ++state) {
      if (bank_for_state[state] == bank && state_count < 4) {
        states[state_count++] = state;
      }
    }
    float local_error[4] = {INFINITY, INFINITY, INFINITY, INFINITY};
    uint local_index[4] = {0, 0, 0, 0};
    for (uint entry = lane; entry < entries; entry += 32) {
      float signal = 0.0f;
      float dot = 0.0f;
      float norm = 0.0f;
      for (int coordinate = 0; coordinate < valid_vector; ++coordinate) {
        const uint position = first + vector * 8 + uint(coordinate);
        const uint offset = row * padded_width + position;
        const float source = value[offset];
        const float objective = weight[offset];
        const float code = float(codebooks[
            (bank * 8 + uint(coordinate)) * entries + entry]);
        signal = fma(objective * source, source, signal);
        dot = fma(objective * source, code, dot);
        norm = fma(objective * code, code, norm);
      }
      for (uint rank = 0; rank < 4; ++rank) {
        const uint state = states[rank];
        const float scale = anchor[row] * scale_lut[state];
        const float error = fma(scale * scale, norm,
                                fma(-2.0f * scale, dot, signal));
        if (error < local_error[rank] ||
            (error == local_error[rank] && entry < local_index[rank])) {
          local_error[rank] = error;
          local_index[rank] = entry;
        }
      }
    }
    for (uint rank = 0; rank < 4; ++rank) {
      for (uint offset = 16; offset > 0; offset >>= 1) {
        const float other_error = simd_shuffle_down(local_error[rank], offset);
        const uint other_index = simd_shuffle_down(local_index[rank], offset);
        if (lane < offset &&
            (other_error < local_error[rank] ||
             (other_error == local_error[rank] &&
              other_index < local_index[rank]))) {
          local_error[rank] = other_error;
          local_index[rank] = other_index;
        }
      }
      if (lane == 0) {
        const uint state = states[rank];
        vector_error[state * 3 + vector] = local_error[rank];
        vector_index[state * 3 + vector] = local_index[rank];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float best_error = INFINITY;
    uint best_state = 0;
    uint best_indices[3] = {0, 0, 0};
    for (uint state = 0; state < 16; ++state) {
      const float error = vector_error[state * 3]
                        + vector_error[state * 3 + 1]
                        + vector_error[state * 3 + 2];
      if (error < best_error || (error == best_error && state < best_state)) {
        best_error = error;
        best_state = state;
        for (uint vector_id = 0; vector_id < 3; ++vector_id) {
          best_indices[vector_id] = vector_index[state * 3 + vector_id];
        }
      }
    }
    out_state[flat_group] = uchar(best_state);
    for (uint vector_id = 0; vector_id < 3; ++vector_id) {
      out_indices[flat_group * 3 + vector_id] = IndexT(best_indices[vector_id]);
    }
  }
}

kernel void nvq2j_assign_u8(
    device const float* value,
    device const float* weight,
    device const float* anchor,
    device const float* scale_lut,
    device const uchar* bank_for_state,
    device const char* codebooks,
    device uchar* out_state,
    device uchar* out_indices,
    constant uint& padded_width,
    constant uint& valid_width,
    constant uint& groups_per_row,
    constant uint& entries,
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup float vector_error[16 * 3];
  threadgroup uint vector_index[16 * 3];
  nvq2j_assign_impl(value, weight, anchor, scale_lut, bank_for_state,
                    codebooks, out_state, out_indices, vector_error,
                    vector_index, padded_width,
                    valid_width, groups_per_row, entries, tid, group);
}

kernel void nvq2j_assign_i32(
    device const float* value,
    device const float* weight,
    device const float* anchor,
    device const float* scale_lut,
    device const uchar* bank_for_state,
    device const char* codebooks,
    device uchar* out_state,
    device int* out_indices,
    constant uint& padded_width,
    constant uint& valid_width,
    constant uint& groups_per_row,
    constant uint& entries,
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup float vector_error[16 * 3];
  threadgroup uint vector_index[16 * 3];
  nvq2j_assign_impl(value, weight, anchor, scale_lut, bank_for_state,
                    codebooks, out_state, out_indices, vector_error,
                    vector_index, padded_width,
                    valid_width, groups_per_row, entries, tid, group);
}

template <typename IndexT>
inline void nvq3j_assign_impl(
    device const float* value,
    device const float* weight,
    device const float* anchor,
    device const float* scale_lut,
    device const uchar* bank_for_state,
    device const char* codebooks,
    device uchar* out_state,
    device IndexT* out_indices,
    threadgroup float* vector_error,
    threadgroup uint* vector_index,
    constant uint& padded_width,
    constant uint& valid_width,
    constant uint& groups_per_row,
    constant uint& entries,
    uint tid,
    uint flat_group) {
  const uint row = flat_group / groups_per_row;
  const uint group = flat_group - row * groups_per_row;
  const uint vector = tid >> 5;
  const uint lane = tid & 31;
  const uint first = group * 24;
  const uint valid_group = min(uint(24), valid_width - first);
  const int valid_vector = max(0, min(4, int(valid_group) - int(vector * 4)));
  for (uint bank = 0; bank < 2; ++bank) {
    uint states[8];
    uint state_count = 0;
    for (uint state = 0; state < 16; ++state) {
      if (bank_for_state[state] == bank && state_count < 8) {
        states[state_count++] = state;
      }
    }
    float local_error[8] = {
        INFINITY, INFINITY, INFINITY, INFINITY,
        INFINITY, INFINITY, INFINITY, INFINITY};
    uint local_index[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint entry = lane; entry < entries; entry += 32) {
      float signal = 0.0f;
      float dot = 0.0f;
      float norm = 0.0f;
      for (int coordinate = 0; coordinate < valid_vector; ++coordinate) {
        const uint position = first + vector * 4 + uint(coordinate);
        const uint offset = row * padded_width + position;
        const float source = value[offset];
        const float objective = weight[offset];
        const float code = float(codebooks[
            (bank * 4 + uint(coordinate)) * entries + entry]);
        signal = fma(objective * source, source, signal);
        dot = fma(objective * source, code, dot);
        norm = fma(objective * code, code, norm);
      }
      for (uint rank = 0; rank < 8; ++rank) {
        const uint state = states[rank];
        const float scale = anchor[row] * scale_lut[state];
        const float error = fma(scale * scale, norm,
                                fma(-2.0f * scale, dot, signal));
        if (error < local_error[rank] ||
            (error == local_error[rank] && entry < local_index[rank])) {
          local_error[rank] = error;
          local_index[rank] = entry;
        }
      }
    }
    for (uint rank = 0; rank < 8; ++rank) {
      for (uint offset = 16; offset > 0; offset >>= 1) {
        const float other_error = simd_shuffle_down(local_error[rank], offset);
        const uint other_index = simd_shuffle_down(local_index[rank], offset);
        if (lane < offset &&
            (other_error < local_error[rank] ||
             (other_error == local_error[rank] &&
              other_index < local_index[rank]))) {
          local_error[rank] = other_error;
          local_index[rank] = other_index;
        }
      }
      if (lane == 0) {
        const uint state = states[rank];
        vector_error[state * 6 + vector] = local_error[rank];
        vector_index[state * 6 + vector] = local_index[rank];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float best_error = INFINITY;
    uint best_state = 0;
    uint best_indices[6] = {0, 0, 0, 0, 0, 0};
    for (uint state = 0; state < 16; ++state) {
      float error = 0.0f;
      for (uint vector_id = 0; vector_id < 6; ++vector_id) {
        error += vector_error[state * 6 + vector_id];
      }
      if (error < best_error || (error == best_error && state < best_state)) {
        best_error = error;
        best_state = state;
        for (uint vector_id = 0; vector_id < 6; ++vector_id) {
          best_indices[vector_id] = vector_index[state * 6 + vector_id];
        }
      }
    }
    out_state[flat_group] = uchar(best_state);
    for (uint vector_id = 0; vector_id < 6; ++vector_id) {
      out_indices[flat_group * 6 + vector_id] =
          IndexT(best_indices[vector_id]);
    }
  }
}

kernel void nvq3j_assign_u8(
    device const float* value,
    device const float* weight,
    device const float* anchor,
    device const float* scale_lut,
    device const uchar* bank_for_state,
    device const char* codebooks,
    device uchar* out_state,
    device uchar* out_indices,
    constant uint& padded_width,
    constant uint& valid_width,
    constant uint& groups_per_row,
    constant uint& entries,
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup float vector_error[16 * 6];
  threadgroup uint vector_index[16 * 6];
  nvq3j_assign_impl(value, weight, anchor, scale_lut, bank_for_state,
                    codebooks, out_state, out_indices, vector_error,
                    vector_index, padded_width, valid_width,
                    groups_per_row, entries, tid, group);
}

kernel void nvq3j_assign_i32(
    device const float* value,
    device const float* weight,
    device const float* anchor,
    device const float* scale_lut,
    device const uchar* bank_for_state,
    device const char* codebooks,
    device uchar* out_state,
    device int* out_indices,
    constant uint& padded_width,
    constant uint& valid_width,
    constant uint& groups_per_row,
    constant uint& entries,
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup float vector_error[16 * 6];
  threadgroup uint vector_index[16 * 6];
  nvq3j_assign_impl(value, weight, anchor, scale_lut, bank_for_state,
                    codebooks, out_state, out_indices, vector_error,
                    vector_index, padded_width, valid_width,
                    groups_per_row, entries, tid, group);
}
"""


@lru_cache(maxsize=1)
def _library():
    if not torch.backends.mps.is_available():
        raise RuntimeError("NVQ-JSC Metal assignment requires MPS")
    return torch.mps.compile_shader(_SOURCE)


def nvq2j_assign(
    value: torch.Tensor,
    objective_weight: torch.Tensor,
    anchor: torch.Tensor,
    scale_lut: torch.Tensor,
    bank_for_state: torch.Tensor,
    codebooks: torch.Tensor,
    valid_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign fixed NVQ2J states and indices with one Metal dispatch."""

    if value.device.type != "mps" or value.dtype != torch.float32:
        raise TypeError("NVQ2J Metal assignment expects MPS float32 values")
    rows, padded_width = map(int, value.shape)
    if not 0 < int(valid_width) <= padded_width:
        raise ValueError("invalid NVQ2J Metal assignment width")
    groups_per_row = padded_width // 24
    entries = int(codebooks.shape[1])
    states = torch.empty(
        (rows, groups_per_row), device="mps", dtype=torch.uint8
    )
    index_dtype = torch.uint8 if entries == 256 else torch.int32
    indices = torch.empty(
        (rows, groups_per_row, 3), device="mps", dtype=index_dtype
    )
    kernel = (
        _library().nvq2j_assign_u8
        if entries == 256
        else _library().nvq2j_assign_i32
    )
    groups = rows * groups_per_row
    kernel(
        value.contiguous(),
        objective_weight.contiguous(),
        anchor.contiguous(),
        scale_lut.contiguous(),
        bank_for_state.to(torch.uint8).contiguous(),
        codebooks.permute(0, 2, 1).contiguous(),
        states,
        indices,
        padded_width,
        int(valid_width),
        groups_per_row,
        entries,
        threads=groups * 96,
        group_size=96,
    )
    return states, indices


def nvq3j_assign(
    value: torch.Tensor,
    objective_weight: torch.Tensor,
    anchor: torch.Tensor,
    scale_lut: torch.Tensor,
    bank_for_state: torch.Tensor,
    codebooks: torch.Tensor,
    valid_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign fixed NVQ3J states and indices in one Metal dispatch."""

    if value.device.type != "mps" or value.dtype != torch.float32:
        raise TypeError("NVQ3J Metal assignment expects MPS float32 values")
    rows, padded_width = map(int, value.shape)
    if not 0 < int(valid_width) <= padded_width or padded_width % 24:
        raise ValueError("invalid NVQ3J Metal assignment width")
    entries = int(codebooks.shape[1])
    if tuple(codebooks.shape) != (2, entries, 4) or entries not in {
        256,
        512,
        1024,
    }:
        raise ValueError(
            "NVQ3J Metal assignment requires [2,256|512|1024,4] tables"
        )
    groups_per_row = padded_width // 24
    states = torch.empty(
        (rows, groups_per_row), device="mps", dtype=torch.uint8
    )
    index_dtype = torch.uint8 if entries == 256 else torch.int32
    indices = torch.empty(
        (rows, groups_per_row, 6), device="mps", dtype=index_dtype
    )
    groups = rows * groups_per_row
    kernel = (
        _library().nvq3j_assign_u8
        if entries == 256
        else _library().nvq3j_assign_i32
    )
    kernel(
        value.contiguous(),
        objective_weight.contiguous(),
        anchor.contiguous(),
        scale_lut.contiguous(),
        bank_for_state.to(torch.uint8).contiguous(),
        codebooks.permute(0, 2, 1).contiguous(),
        states,
        indices,
        padded_width,
        int(valid_width),
        groups_per_row,
        entries,
        threads=groups * 192,
        group_size=192,
    )
    return states, indices


__all__ = ["nvq2j_assign", "nvq3j_assign"]
