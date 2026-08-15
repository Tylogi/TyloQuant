"""Metal projection kernel for the 256-bank NEPQ0-S quantizer."""

from __future__ import annotations

from functools import lru_cache

import torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

inline bool better(float error, uint index, float best_error, uint best_index) {
  return error < best_error || (error == best_error && index < best_index);
}

inline uint table_offset(uint state, uint entry, uint coordinate, uint bank) {
  return (((state * 8 + entry) * 4 + coordinate) * 256) + bank;
}

kernel void nepq0_s_assign(
    device const float* value,
    device const float* anchor,
    device const float* scale_lut,
    device const char* first_tables,
    device const char* second_tables,
    device uchar* out_bank,
    device uchar* out_state,
    device uchar* out_indices,
    constant uint& width,
    constant uint& groups_per_row,
    constant uint& supers_per_row,
    uint bank [[thread_index_in_threadgroup]],
    uint flat_super [[threadgroup_position_in_grid]]) {
  threadgroup float shared_value[96];
  threadgroup float bank_error[256];
  threadgroup uint winning_bank;
  const uint row = flat_super / supers_per_row;
  const uint super_index = flat_super - row * supers_per_row;
  const uint first_group = super_index * 4;
  const uint first_position = first_group * 24;
  if (bank < 96) {
    const uint position = first_position + bank;
    shared_value[bank] = position < width
        ? value[row * width + position]
        : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float super_error = 0.0f;
  uchar selected_state[4] = {0, 0, 0, 0};
  uchar selected_indices[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const float row_anchor = anchor[row];
  for (uint local_group = 0; local_group < 4; ++local_group) {
    const uint group = first_group + local_group;
    if (group >= groups_per_row) {
      continue;
    }
    const uint valid_group = min(uint(24), width - group * 24);
    float best_group_error = INFINITY;
    uint best_group_state = 0;
    uchar best_group_indices[3] = {0, 0, 0};
    for (uint state = 0; state < 4; ++state) {
      const float scale = row_anchor * scale_lut[state * 256 + bank];
      float group_error = 0.0f;
      uchar group_indices[3] = {0, 0, 0};
      for (uint vector = 0; vector < 3; ++vector) {
        const int valid_vector = max(
            0, min(8, int(valid_group) - int(vector * 8)));
        uint composite = 0;
        for (uint half_id = 0; half_id < 2; ++half_id) {
          const int valid_half = half_id == 0
              ? min(4, valid_vector)
              : max(0, valid_vector - 4);
          float best_error = valid_half > 0 ? INFINITY : 0.0f;
          uint best_index = 0;
          if (valid_half > 0) {
            device const char* table = half_id == 0
                ? first_tables : second_tables;
            for (uint entry = 0; entry < 8; ++entry) {
              float error = 0.0f;
              for (int coordinate = 0; coordinate < valid_half; ++coordinate) {
                const uint position = local_group * 24 + vector * 8
                                    + half_id * 4 + uint(coordinate);
                const float source = shared_value[position];
                const float code = float(table[
                    table_offset(state, entry, uint(coordinate), bank)]);
                const float residual = source - scale * code;
                error = fma(residual, residual, error);
              }
              if (better(error, entry, best_error, best_index)) {
                best_error = error;
                best_index = entry;
              }
            }
          }
          group_error += best_error;
          composite |= best_index << (half_id * 3);
        }
        group_indices[vector] = uchar(composite);
      }
      if (better(group_error, state, best_group_error, best_group_state)) {
        best_group_error = group_error;
        best_group_state = state;
        for (uint vector = 0; vector < 3; ++vector) {
          best_group_indices[vector] = group_indices[vector];
        }
      }
    }
    super_error += best_group_error;
    selected_state[local_group] = uchar(best_group_state);
    for (uint vector = 0; vector < 3; ++vector) {
      selected_indices[local_group * 3 + vector] = best_group_indices[vector];
    }
  }
  bank_error[bank] = super_error;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (bank == 0) {
    float best_error = bank_error[0];
    uint best_bank = 0;
    for (uint candidate = 1; candidate < 256; ++candidate) {
      if (better(bank_error[candidate], candidate, best_error, best_bank)) {
        best_error = bank_error[candidate];
        best_bank = candidate;
      }
    }
    winning_bank = best_bank;
    out_bank[row * supers_per_row + super_index] = uchar(best_bank);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (bank == winning_bank) {
    for (uint local_group = 0; local_group < 4; ++local_group) {
      const uint group = first_group + local_group;
      if (group >= groups_per_row) {
        continue;
      }
      const uint group_offset = row * groups_per_row + group;
      out_state[group_offset] = selected_state[local_group];
      for (uint vector = 0; vector < 3; ++vector) {
        out_indices[group_offset * 3 + vector] =
            selected_indices[local_group * 3 + vector];
      }
    }
  }
}

kernel void nepq0_s_refit_anchor(
    device const float* value,
    device const float* previous_anchor,
    device const float* scale_lut,
    device const char* first_tables,
    device const char* second_tables,
    device const uchar* bank_ids,
    device const uchar* states,
    device const uchar* indices,
    device float* out_anchor,
    constant uint& width,
    constant uint& groups_per_row,
    constant uint& supers_per_row,
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  threadgroup float numerator_parts[256];
  threadgroup float denominator_parts[256];
  float numerator = 0.0f;
  float denominator = 0.0f;
  for (uint position = tid; position < width; position += 256) {
    const uint group = position / 24;
    const uint in_group = position - group * 24;
    const uint vector = in_group / 8;
    const uint in_vector = in_group - vector * 8;
    const uint bank = bank_ids[
        row * supers_per_row + group / 4];
    const uint state = states[row * groups_per_row + group];
    const uint composite = indices[(row * groups_per_row + group) * 3 + vector];
    const bool second = in_vector >= 4;
    const uint entry = second ? composite >> 3 : composite & 7;
    const uint coordinate = second ? in_vector - 4 : in_vector;
    device const char* table = second ? second_tables : first_tables;
    const float code = float(table[
        table_offset(state, entry, coordinate, bank)]);
    const float basis = scale_lut[state * 256 + bank] * code;
    const float source = value[row * width + position];
    numerator = fma(source, basis, numerator);
    denominator = fma(basis, basis, denominator);
  }
  numerator_parts[tid] = numerator;
  denominator_parts[tid] = denominator;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      numerator_parts[tid] += numerator_parts[tid + stride];
      denominator_parts[tid] += denominator_parts[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    const float fitted = denominator_parts[0] > 0.0f
        ? max(numerator_parts[0] / denominator_parts[0], 0.0f)
        : previous_anchor[row];
    out_anchor[row] = float(half(fitted));
  }
}
"""


@lru_cache(maxsize=1)
def _library():
    if not torch.backends.mps.is_available():
        raise RuntimeError("NEPQ0-S Metal assignment requires MPS")
    return torch.mps.compile_shader(_SOURCE)


def _dispatch_assignment(
    value: torch.Tensor,
    anchor: torch.Tensor,
    scale_lut: torch.Tensor,
    first_tables: torch.Tensor,
    second_tables: torch.Tensor,
    bank_ids: torch.Tensor,
    states: torch.Tensor,
    indices: torch.Tensor,
    groups_per_row: int,
    supers_per_row: int,
) -> None:
    rows, width = map(int, value.shape)
    groups = rows * supers_per_row
    _library().nepq0_s_assign(
        value,
        anchor,
        scale_lut,
        first_tables,
        second_tables,
        bank_ids,
        states,
        indices,
        width,
        groups_per_row,
        supers_per_row,
        threads=groups * 256,
        group_size=256,
    )


def nepq0_s_assign(
    value: torch.Tensor,
    initial_anchor: torch.Tensor,
    scale_lut: torch.Tensor,
    first_tables: torch.Tensor,
    second_tables: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project rows into a 256-bank NEPQ0-S pool, refit, then reassign."""

    if value.device.type != "mps" or value.dtype != torch.float32:
        raise TypeError("NEPQ0-S Metal assignment expects MPS float32 values")
    rows, width = map(int, value.shape)
    if rows <= 0 or width <= 0 or width % 8:
        raise ValueError("NEPQ0-S Metal values require nonempty [rows,K], K % 8 == 0")
    if tuple(scale_lut.shape) != (4, 256):
        raise ValueError("NEPQ0-S Metal scale table must have shape [4,256]")
    if tuple(first_tables.shape) != (4, 8, 4, 256):
        raise ValueError("NEPQ0-S Metal first tables must have shape [4,8,4,256]")
    if tuple(second_tables.shape) != (4, 8, 4, 256):
        raise ValueError("NEPQ0-S Metal second tables must have shape [4,8,4,256]")
    value = value.contiguous()
    initial_anchor = initial_anchor.contiguous()
    scale_lut = scale_lut.contiguous()
    first_tables = first_tables.contiguous()
    second_tables = second_tables.contiguous()
    groups_per_row = (width + 23) // 24
    supers_per_row = (groups_per_row + 3) // 4
    bank_ids = torch.empty(
        (rows, supers_per_row), device="mps", dtype=torch.uint8
    )
    states = torch.empty(
        (rows, groups_per_row), device="mps", dtype=torch.uint8
    )
    indices = torch.empty(
        (rows, groups_per_row, 3), device="mps", dtype=torch.uint8
    )
    _dispatch_assignment(
        value, initial_anchor, scale_lut, first_tables, second_tables,
        bank_ids, states, indices, groups_per_row, supers_per_row
    )
    fitted_anchor = torch.empty_like(initial_anchor)
    _library().nepq0_s_refit_anchor(
        value,
        initial_anchor,
        scale_lut,
        first_tables,
        second_tables,
        bank_ids,
        states,
        indices,
        fitted_anchor,
        width,
        groups_per_row,
        supers_per_row,
        threads=rows * 256,
        group_size=256,
    )
    _dispatch_assignment(
        value, fitted_anchor, scale_lut, first_tables, second_tables,
        bank_ids, states, indices, groups_per_row, supers_per_row
    )
    return fitted_anchor, bank_ids, states, indices


__all__ = ["nepq0_s_assign"]
