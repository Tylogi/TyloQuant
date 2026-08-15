"""Metal pooled-table assignment for general NEPQ0-S and NEPQ0-L."""

from __future__ import annotations

from functools import lru_cache

import torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

inline bool lower(float error, uint index, float best_error, uint best_index) {
  return error < best_error || (error == best_error && index < best_index);
}

kernel void nepq0_candidates(
    device const float* value,
    device const float* weight,
    device const float* anchor,
    device const float* scale_lut,
    device const float* first_codebooks,
    device const float* second_codebooks,
    device float* candidate_error,
    device uchar* candidate_state,
    device int* candidate_indices,
    constant uint& padded_width,
    constant uint& valid_width,
    constant uint& groups_per_row,
    constant uint& supers_per_row,
    constant uint& bank_count,
    constant uint& state_count,
    constant uint& first_entries,
    constant uint& second_entries,
    uint tid [[thread_index_in_threadgroup]],
    uint flat_candidate [[threadgroup_position_in_grid]]) {
  threadgroup float vector_error[8 * 3];
  threadgroup uint vector_first[8 * 3];
  threadgroup uint vector_second[8 * 3];
  const uint bank = flat_candidate % bank_count;
  const uint flat_super = flat_candidate / bank_count;
  const uint row = flat_super / supers_per_row;
  const uint super_index = flat_super - row * supers_per_row;
  const uint first_group = super_index * 4;
  const uint vector = tid >> 5;
  const uint lane = tid & 31;
  const uint first_bank_stride = state_count * first_entries * 4;
  const uint second_bank_stride = state_count * second_entries * 4;
  float super_error = 0.0f;

  for (uint local_group = 0; local_group < 4; ++local_group) {
    const uint group = first_group + local_group;
    if (group >= groups_per_row) {
      continue;
    }
    const uint first_position = group * 24;
    const uint valid_group = min(uint(24), valid_width - first_position);
    const int valid_vector = max(
        0, min(8, int(valid_group) - int(vector * 8)));
    const int valid_first = min(4, valid_vector);
    const int valid_second = max(0, valid_vector - 4);
    for (uint state = 0; state < state_count; ++state) {
      const float scale = anchor[row]
                        * scale_lut[bank * state_count + state];
      float first_error = INFINITY;
      uint first_index = 0;
      if (lane < first_entries) {
        first_index = lane;
        float signal = 0.0f;
        float dot = 0.0f;
        float norm = 0.0f;
        for (int coordinate = 0; coordinate < valid_first; ++coordinate) {
          const uint position = first_position + vector * 8 + uint(coordinate);
          const uint offset = row * padded_width + position;
          const float code = first_codebooks[
              bank * first_bank_stride
              + (state * first_entries + lane) * 4 + uint(coordinate)];
          const float source = value[offset];
          const float objective = weight[offset];
          signal = fma(objective * source, source, signal);
          dot = fma(objective * source, code, dot);
          norm = fma(objective * code, code, norm);
        }
        first_error = fma(
            scale * scale, norm, fma(-2.0f * scale, dot, signal));
      }
      for (uint offset = 16; offset > 0; offset >>= 1) {
        const float other_error = simd_shuffle_down(first_error, offset);
        const uint other_index = simd_shuffle_down(first_index, offset);
        if (lane < offset && lower(
                other_error, other_index, first_error, first_index)) {
          first_error = other_error;
          first_index = other_index;
        }
      }

      float second_error = INFINITY;
      uint second_index = 0;
      if (lane < second_entries) {
        second_index = lane;
        float signal = 0.0f;
        float dot = 0.0f;
        float norm = 0.0f;
        for (int coordinate = 0; coordinate < valid_second; ++coordinate) {
          const uint position = first_position + vector * 8 + 4 + uint(coordinate);
          const uint offset = row * padded_width + position;
          const float code = second_codebooks[
              bank * second_bank_stride
              + (state * second_entries + lane) * 4 + uint(coordinate)];
          const float source = value[offset];
          const float objective = weight[offset];
          signal = fma(objective * source, source, signal);
          dot = fma(objective * source, code, dot);
          norm = fma(objective * code, code, norm);
        }
        second_error = fma(
            scale * scale, norm, fma(-2.0f * scale, dot, signal));
      }
      for (uint offset = 16; offset > 0; offset >>= 1) {
        const float other_error = simd_shuffle_down(second_error, offset);
        const uint other_index = simd_shuffle_down(second_index, offset);
        if (lane < offset && lower(
                other_error, other_index, second_error, second_index)) {
          second_error = other_error;
          second_index = other_index;
        }
      }
      if (lane == 0) {
        const uint result = state * 3 + vector;
        vector_error[result] = first_error + second_error;
        vector_first[result] = first_index;
        vector_second[result] = second_index;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float best_group_error = INFINITY;
      uint best_state = 0;
      uint best_indices[3] = {0, 0, 0};
      for (uint state = 0; state < state_count; ++state) {
        float error = 0.0f;
        for (uint vector_id = 0; vector_id < 3; ++vector_id) {
          error += vector_error[state * 3 + vector_id];
        }
        if (error < best_group_error) {
          best_group_error = error;
          best_state = state;
          for (uint vector_id = 0; vector_id < 3; ++vector_id) {
            best_indices[vector_id] =
                vector_first[state * 3 + vector_id]
                | (vector_second[state * 3 + vector_id] << 3);
          }
        }
      }
      super_error += best_group_error;
      const uint candidate_group =
          (flat_super * bank_count + bank) * 4 + local_group;
      candidate_state[candidate_group] = uchar(best_state);
      for (uint vector_id = 0; vector_id < 3; ++vector_id) {
        candidate_indices[candidate_group * 3 + vector_id] =
            int(best_indices[vector_id]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    candidate_error[flat_candidate] = super_error;
  }
}

kernel void nepq0_select_banks(
    device const float* candidate_error,
    device const uchar* candidate_state,
    device const int* candidate_indices,
    device uchar* out_bank,
    device uchar* out_state,
    device int* out_indices,
    device float* out_super_error,
    constant uint& groups_per_row,
    constant uint& supers_per_row,
    constant uint& bank_count,
    uint tid [[thread_index_in_threadgroup]],
    uint flat_super [[threadgroup_position_in_grid]]) {
  threadgroup float error_parts[256];
  threadgroup uint bank_parts[256];
  error_parts[tid] = tid < bank_count
      ? candidate_error[flat_super * bank_count + tid]
      : INFINITY;
  bank_parts[tid] = tid;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride && lower(
            error_parts[tid + stride], bank_parts[tid + stride],
            error_parts[tid], bank_parts[tid])) {
      error_parts[tid] = error_parts[tid + stride];
      bank_parts[tid] = bank_parts[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    const uint bank = bank_parts[0];
    const uint row = flat_super / supers_per_row;
    const uint super_index = flat_super - row * supers_per_row;
    const uint first_group = super_index * 4;
    out_bank[flat_super] = uchar(bank);
    out_super_error[flat_super] = error_parts[0];
    for (uint local_group = 0; local_group < 4; ++local_group) {
      const uint group = first_group + local_group;
      if (group >= groups_per_row) {
        continue;
      }
      const uint candidate_group =
          (flat_super * bank_count + bank) * 4 + local_group;
      const uint output_group = row * groups_per_row + group;
      out_state[output_group] = candidate_state[candidate_group];
      for (uint vector = 0; vector < 3; ++vector) {
        out_indices[output_group * 3 + vector] =
            candidate_indices[candidate_group * 3 + vector];
      }
    }
  }
}
"""


@lru_cache(maxsize=1)
def _library():
    if not torch.backends.mps.is_available():
        raise RuntimeError("NEPQ0 pooled Metal assignment requires MPS")
    return torch.mps.compile_shader(_SOURCE)


def assign_nepq0_pool(
    value: torch.Tensor,
    objective: torch.Tensor,
    anchor: torch.Tensor,
    scale_lut: torch.Tensor,
    first_codebooks: torch.Tensor,
    second_codebooks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign arbitrary NEPQ0-S/L table pools with weighted SSE."""

    if value.device.type != "mps" or value.dtype != torch.float32:
        raise TypeError("NEPQ0 pooled Metal assignment expects MPS float32")
    rows, valid_width = map(int, value.shape)
    bank_count, state_count = map(int, scale_lut.shape)
    if not 1 <= bank_count <= 256 or state_count not in {4, 8}:
        raise ValueError("invalid NEPQ0 Metal pool geometry")
    if tuple(first_codebooks.shape[:2]) != (bank_count, state_count):
        raise ValueError("NEPQ0 Metal first tables do not match scale tables")
    if tuple(second_codebooks.shape[:2]) != (bank_count, state_count):
        raise ValueError("NEPQ0 Metal second tables do not match scale tables")
    first_entries = int(first_codebooks.shape[2])
    second_entries = int(second_codebooks.shape[2])
    if first_entries != 8 or second_entries not in {8, 16}:
        raise ValueError("unsupported NEPQ0 Metal product-codebook sizes")
    groups_per_row = (valid_width + 23) // 24
    padded_width = groups_per_row * 24
    if padded_width != valid_width:
        value = torch.nn.functional.pad(value, (0, padded_width - valid_width))
        objective = torch.nn.functional.pad(
            objective, (0, padded_width - valid_width)
        )
    supers_per_row = (groups_per_row + 3) // 4
    total_supers = rows * supers_per_row
    total_candidates = total_supers * bank_count
    candidate_error = torch.empty(
        total_candidates, device="mps", dtype=torch.float32
    )
    candidate_state = torch.empty(
        (total_candidates, 4), device="mps", dtype=torch.uint8
    )
    candidate_indices = torch.empty(
        (total_candidates, 4, 3), device="mps", dtype=torch.int32
    )
    _library().nepq0_candidates(
        value.contiguous(),
        objective.contiguous(),
        anchor.contiguous(),
        scale_lut.contiguous(),
        first_codebooks.contiguous(),
        second_codebooks.contiguous(),
        candidate_error,
        candidate_state,
        candidate_indices,
        padded_width,
        valid_width,
        groups_per_row,
        supers_per_row,
        bank_count,
        state_count,
        first_entries,
        second_entries,
        threads=total_candidates * 96,
        group_size=96,
    )
    bank_ids = torch.empty(
        (rows, supers_per_row), device="mps", dtype=torch.uint8
    )
    state = torch.empty(
        (rows, groups_per_row), device="mps", dtype=torch.uint8
    )
    indices_i32 = torch.empty(
        (rows, groups_per_row, 3), device="mps", dtype=torch.int32
    )
    super_error = torch.empty(
        (rows, supers_per_row), device="mps", dtype=torch.float32
    )
    _library().nepq0_select_banks(
        candidate_error,
        candidate_state,
        candidate_indices,
        bank_ids,
        state,
        indices_i32,
        super_error,
        groups_per_row,
        supers_per_row,
        bank_count,
        threads=total_supers * 256,
        group_size=256,
    )
    return bank_ids, state, indices_i32.to(torch.int64), super_error.sum(1)


__all__ = ["assign_nepq0_pool"]
