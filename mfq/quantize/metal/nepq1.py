"""Metal pooled-table assignment for NEPQ1-S and NEPQ1-L."""

from __future__ import annotations

from functools import lru_cache

import torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

inline bool lower(float error, uint index, float best_error, uint best_index) {
  return error < best_error || (error == best_error && index < best_index);
}

kernel void nepq1_candidates(
    device const float* value,
    device const float* weight,
    device const float* anchor,
    device const float* codebooks,
    device float* candidate_error,
    device uchar* candidate_state,
    device uchar* candidate_aux,
    device int* candidate_indices,
    constant uint& padded_width,
    constant uint& valid_width,
    constant uint& groups_per_row,
    constant uint& supers_per_row,
    constant uint& bank_count,
    constant uint& entries,
    constant uint& state_count,
    constant uint& bank_stride,
    constant uint& aux_stride,
    constant float& delta_value,
    uint tid [[thread_index_in_threadgroup]],
    uint flat_candidate [[threadgroup_position_in_grid]]) {
  threadgroup float vector_error[2 * 16 * 3];
  threadgroup uint vector_index[2 * 16 * 3];
  const uint bank = flat_candidate % bank_count;
  const uint flat_super = flat_candidate / bank_count;
  const uint row = flat_super / supers_per_row;
  const uint super_index = flat_super - row * supers_per_row;
  const uint first_group = super_index * 4;
  const uint vector = tid >> 5;
  const uint lane = tid & 31;
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
    for (uint aux = 0; aux < 2; ++aux) {
      const float delta = aux == 0 ? delta_value : -delta_value;
      const uint codebook_base = bank * bank_stride + aux * aux_stride;
      for (uint state = 0; state < state_count; ++state) {
        float best_error = INFINITY;
        uint best_index = 0;
        for (uint entry = lane; entry < entries; entry += 32) {
          float dot = 0.0f;
          float norm = 0.0f;
          for (int coordinate = 0; coordinate < valid_vector; ++coordinate) {
            const uint position = first_position + vector * 8 + uint(coordinate);
            const uint offset = row * padded_width + position;
            const float code = codebooks[
                codebook_base + entry * 8 + uint(coordinate)] + delta;
            const float source = value[offset];
            const float objective = weight[offset];
            dot = fma(objective * source, code, dot);
            norm = fma(objective * code, code, norm);
          }
          const float scale = anchor[row] * float(state);
          const float error = fma(scale * scale, norm, -2.0f * scale * dot);
          if (lower(error, entry, best_error, best_index)) {
            best_error = error;
            best_index = entry;
          }
        }
        for (uint offset = 16; offset > 0; offset >>= 1) {
          const float other_error = simd_shuffle_down(best_error, offset);
          const uint other_index = simd_shuffle_down(best_index, offset);
          if (lane < offset && lower(
                  other_error, other_index, best_error, best_index)) {
            best_error = other_error;
            best_index = other_index;
          }
        }
        if (lane == 0) {
          const uint result = (aux * 16 + state) * 3 + vector;
          vector_error[result] = best_error;
          vector_index[result] = best_index;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float best_group_error = INFINITY;
      uint best_state = 0;
      uint best_aux = 0;
      uint best_indices[3] = {0, 0, 0};
      for (uint aux = 0; aux < 2; ++aux) {
        for (uint state = 0; state < state_count; ++state) {
          float error = 0.0f;
          for (uint vector_id = 0; vector_id < 3; ++vector_id) {
            error += vector_error[(aux * 16 + state) * 3 + vector_id];
          }
          if (error < best_group_error) {
            best_group_error = error;
            best_state = state;
            best_aux = aux;
            for (uint vector_id = 0; vector_id < 3; ++vector_id) {
              best_indices[vector_id] =
                  vector_index[(aux * 16 + state) * 3 + vector_id];
            }
          }
        }
      }
      super_error += best_group_error;
      const uint candidate_group =
          (flat_super * bank_count + bank) * 4 + local_group;
      candidate_state[candidate_group] = uchar(best_state);
      candidate_aux[candidate_group] = uchar(best_aux);
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

kernel void nepq1_select_banks(
    device const float* candidate_error,
    device const uchar* candidate_state,
    device const uchar* candidate_aux,
    device const int* candidate_indices,
    device uchar* out_bank,
    device uchar* out_state,
    device uchar* out_aux,
    device int* out_indices,
    device float* out_super_error,
    constant uint& groups_per_row,
    constant uint& supers_per_row,
    constant uint& bank_count,
    uint tid [[thread_index_in_threadgroup]],
    uint flat_super [[threadgroup_position_in_grid]]) {
  threadgroup float error_parts[256];
  threadgroup uint bank_parts[256];
  const float error = tid < bank_count
      ? candidate_error[flat_super * bank_count + tid]
      : INFINITY;
  error_parts[tid] = error;
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
      out_aux[output_group] = candidate_aux[candidate_group];
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
        raise RuntimeError("NEPQ1 Metal assignment requires MPS")
    return torch.mps.compile_shader(_SOURCE)


def assign_nepq1(
    value: torch.Tensor,
    objective: torch.Tensor,
    anchor: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    states: int,
    delta: float,
    banked_codebooks: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign NEPQ1 pooled codebooks without materializing score tensors."""

    if value.device.type != "mps" or value.dtype != torch.float32:
        raise TypeError("NEPQ1 Metal assignment expects MPS float32 values")
    rows, valid_width = map(int, value.shape)
    groups_per_row = (valid_width + 23) // 24
    padded_width = groups_per_row * 24
    if padded_width != valid_width:
        value = torch.nn.functional.pad(value, (0, padded_width - valid_width))
        objective = torch.nn.functional.pad(
            objective, (0, padded_width - valid_width)
        )
    if banked_codebooks:
        bank_count, aux_count, entries, vector_size = map(int, codebooks.shape)
        if aux_count != 2:
            raise ValueError("NEPQ1-S Metal tables require two codebook banks")
        aux_stride = entries * vector_size
        bank_stride = aux_count * aux_stride
    else:
        bank_count, entries, vector_size = map(int, codebooks.shape)
        aux_stride = 0
        bank_stride = entries * vector_size
    if vector_size != 8 or not 1 <= bank_count <= 256:
        raise ValueError("invalid NEPQ1 Metal codebook pool")
    if states not in {8, 16}:
        raise ValueError("NEPQ1 Metal state count must be 8 or 16")
    supers_per_row = (groups_per_row + 3) // 4
    total_supers = rows * supers_per_row
    total_candidates = total_supers * bank_count
    candidate_error = torch.empty(
        total_candidates, device="mps", dtype=torch.float32
    )
    candidate_state = torch.empty(
        (total_candidates, 4), device="mps", dtype=torch.uint8
    )
    candidate_aux = torch.empty_like(candidate_state)
    candidate_indices = torch.empty(
        (total_candidates, 4, 3), device="mps", dtype=torch.int32
    )
    _library().nepq1_candidates(
        value.contiguous(),
        objective.contiguous(),
        anchor.contiguous(),
        codebooks.contiguous(),
        candidate_error,
        candidate_state,
        candidate_aux,
        candidate_indices,
        padded_width,
        valid_width,
        groups_per_row,
        supers_per_row,
        bank_count,
        entries,
        int(states),
        bank_stride,
        aux_stride,
        float(delta),
        threads=total_candidates * 96,
        group_size=96,
    )
    bank_ids = torch.empty(
        (rows, supers_per_row), device="mps", dtype=torch.uint8
    )
    state = torch.empty(
        (rows, groups_per_row), device="mps", dtype=torch.uint8
    )
    aux = torch.empty_like(state)
    indices_i32 = torch.empty(
        (rows, groups_per_row, 3), device="mps", dtype=torch.int32
    )
    super_error = torch.empty(
        (rows, supers_per_row), device="mps", dtype=torch.float32
    )
    _library().nepq1_select_banks(
        candidate_error,
        candidate_state,
        candidate_aux,
        candidate_indices,
        bank_ids,
        state,
        aux,
        indices_i32,
        super_error,
        groups_per_row,
        supers_per_row,
        bank_count,
        threads=total_supers * 256,
        group_size=256,
    )
    return bank_ids, state, indices_i32.to(torch.int64), aux, super_error.sum(1)


__all__ = ["assign_nepq1"]
