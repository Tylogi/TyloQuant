"""Fused Metal assignment kernels for fixed NPQ0-S and NPQ0-L tables."""

from __future__ import annotations

from functools import lru_cache

import torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

inline bool better(float error, uint index, float best_error, uint best_index) {
  return error < best_error || (error == best_error && index < best_index);
}

kernel void npq0_s_assign(
    device const float* value,
    device const float* weight,
    device const float* anchor,
    device const float* scale_lut,
    device const char* first_codebooks,
    device const char* second_codebooks,
    device uchar* out_state,
    device uchar* out_first,
    device uchar* out_second,
    device float* out_group_error,
    constant uint& padded_width,
    constant uint& valid_width,
    constant uint& groups_per_row,
    uint tid [[thread_index_in_threadgroup]],
    uint flat_group [[threadgroup_position_in_grid]]) {
  threadgroup float candidates[4 * 3 * 16];
  const uint row = flat_group / groups_per_row;
  const uint group = flat_group - row * groups_per_row;
  const uint vector = tid >> 5;
  const uint lane = tid & 31;
  const uint first_position = group * 24;
  const uint valid_group = min(uint(24), valid_width - first_position);
  const int valid_vector = max(0, min(8, int(valid_group) - int(vector * 8)));

  if (lane < 16) {
    const bool second = lane >= 8;
    const uint entry = lane & 7;
    const int valid_half = second
        ? max(0, valid_vector - 4)
        : min(4, valid_vector);
    const uint coordinate_base = first_position + vector * 8 + (second ? 4 : 0);
    device const char* codebook = second ? second_codebooks : first_codebooks;
    for (uint state = 0; state < 4; ++state) {
      float signal = 0.0f;
      float dot = 0.0f;
      float norm = 0.0f;
      for (int coordinate = 0; coordinate < valid_half; ++coordinate) {
        const uint offset = row * padded_width + coordinate_base + uint(coordinate);
        const float source = value[offset];
        const float objective = weight[offset];
        const float code = float(codebook[
            (state * 8 + entry) * 4 + uint(coordinate)]);
        signal = fma(objective * source, source, signal);
        dot = fma(objective * source, code, dot);
        norm = fma(objective * code, code, norm);
      }
      const float scale = anchor[row] * scale_lut[state];
      candidates[(state * 3 + vector) * 16 + lane] =
          fma(scale * scale, norm, fma(-2.0f * scale, dot, signal));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float best_group_error = INFINITY;
    uint best_state = 0;
    uint best_first[3] = {0, 0, 0};
    uint best_second[3] = {0, 0, 0};
    for (uint state = 0; state < 4; ++state) {
      float group_error = 0.0f;
      uint state_first[3] = {0, 0, 0};
      uint state_second[3] = {0, 0, 0};
      for (uint vector_id = 0; vector_id < 3; ++vector_id) {
        float first_error = INFINITY;
        float second_error = INFINITY;
        for (uint entry = 0; entry < 8; ++entry) {
          const float candidate_first =
              candidates[(state * 3 + vector_id) * 16 + entry];
          if (better(candidate_first, entry, first_error, state_first[vector_id])) {
            first_error = candidate_first;
            state_first[vector_id] = entry;
          }
          const float candidate_second =
              candidates[(state * 3 + vector_id) * 16 + 8 + entry];
          if (better(candidate_second, entry, second_error, state_second[vector_id])) {
            second_error = candidate_second;
            state_second[vector_id] = entry;
          }
        }
        group_error += first_error + second_error;
      }
      if (better(group_error, state, best_group_error, best_state)) {
        best_group_error = group_error;
        best_state = state;
        for (uint vector_id = 0; vector_id < 3; ++vector_id) {
          best_first[vector_id] = state_first[vector_id];
          best_second[vector_id] = state_second[vector_id];
        }
      }
    }
    out_state[flat_group] = uchar(best_state);
    out_group_error[flat_group] = best_group_error;
    for (uint vector_id = 0; vector_id < 3; ++vector_id) {
      out_first[flat_group * 3 + vector_id] = uchar(best_first[vector_id]);
      out_second[flat_group * 3 + vector_id] = uchar(best_second[vector_id]);
    }
  }
}

kernel void npq0_l_assign(
    device const float* value,
    device const float* weight,
    device const float* anchor,
    device const float* scale_lut,
    device const char* first_codebooks,
    device const char* second_codebooks,
    device uchar* out_state,
    device uchar* out_first,
    device uchar* out_second,
    device float* out_group_error,
    constant uint& padded_width,
    constant uint& valid_width,
    constant uint& groups_per_row,
    uint tid [[thread_index_in_threadgroup]],
    uint flat_group [[threadgroup_position_in_grid]]) {
  threadgroup float first_candidates[8 * 3 * 8];
  threadgroup float second_candidates[8 * 3 * 16];
  const uint row = flat_group / groups_per_row;
  const uint group = flat_group - row * groups_per_row;
  const uint vector = tid >> 5;
  const uint lane = tid & 31;
  const uint first_position = group * 24;
  const uint valid_group = min(uint(24), valid_width - first_position);
  const int valid_vector = max(0, min(8, int(valid_group) - int(vector * 8)));
  const int valid_first = min(4, valid_vector);
  const int valid_second = max(0, valid_vector - 4);

  for (uint state = 0; state < 8; ++state) {
    const float scale = anchor[row] * scale_lut[state];
    if (lane < 8) {
      float signal = 0.0f;
      float dot = 0.0f;
      float norm = 0.0f;
      for (int coordinate = 0; coordinate < valid_first; ++coordinate) {
        const uint offset = row * padded_width + first_position
                          + vector * 8 + uint(coordinate);
        const float source = value[offset];
        const float objective = weight[offset];
        const float code = float(first_codebooks[
            (state * 8 + lane) * 4 + uint(coordinate)]);
        signal = fma(objective * source, source, signal);
        dot = fma(objective * source, code, dot);
        norm = fma(objective * code, code, norm);
      }
      first_candidates[(state * 3 + vector) * 8 + lane] =
          fma(scale * scale, norm, fma(-2.0f * scale, dot, signal));
    }
    if (lane < 16) {
      float signal = 0.0f;
      float dot = 0.0f;
      float norm = 0.0f;
      for (int coordinate = 0; coordinate < valid_second; ++coordinate) {
        const uint offset = row * padded_width + first_position
                          + vector * 8 + 4 + uint(coordinate);
        const float source = value[offset];
        const float objective = weight[offset];
        const float code = float(second_codebooks[
            (state * 16 + lane) * 4 + uint(coordinate)]);
        signal = fma(objective * source, source, signal);
        dot = fma(objective * source, code, dot);
        norm = fma(objective * code, code, norm);
      }
      second_candidates[(state * 3 + vector) * 16 + lane] =
          fma(scale * scale, norm, fma(-2.0f * scale, dot, signal));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float best_group_error = INFINITY;
    uint best_state = 0;
    uint best_first[3] = {0, 0, 0};
    uint best_second[3] = {0, 0, 0};
    for (uint state = 0; state < 8; ++state) {
      float group_error = 0.0f;
      uint state_first[3] = {0, 0, 0};
      uint state_second[3] = {0, 0, 0};
      for (uint vector_id = 0; vector_id < 3; ++vector_id) {
        float first_error = INFINITY;
        float second_error = INFINITY;
        for (uint entry = 0; entry < 8; ++entry) {
          const float candidate =
              first_candidates[(state * 3 + vector_id) * 8 + entry];
          if (better(candidate, entry, first_error, state_first[vector_id])) {
            first_error = candidate;
            state_first[vector_id] = entry;
          }
        }
        for (uint entry = 0; entry < 16; ++entry) {
          const float candidate =
              second_candidates[(state * 3 + vector_id) * 16 + entry];
          if (better(candidate, entry, second_error, state_second[vector_id])) {
            second_error = candidate;
            state_second[vector_id] = entry;
          }
        }
        group_error += first_error + second_error;
      }
      if (better(group_error, state, best_group_error, best_state)) {
        best_group_error = group_error;
        best_state = state;
        for (uint vector_id = 0; vector_id < 3; ++vector_id) {
          best_first[vector_id] = state_first[vector_id];
          best_second[vector_id] = state_second[vector_id];
        }
      }
    }
    out_state[flat_group] = uchar(best_state);
    out_group_error[flat_group] = best_group_error;
    for (uint vector_id = 0; vector_id < 3; ++vector_id) {
      out_first[flat_group * 3 + vector_id] = uchar(best_first[vector_id]);
      out_second[flat_group * 3 + vector_id] = uchar(best_second[vector_id]);
    }
  }
}

kernel void npq0_s_batched_assign(
    device const float* value,
    device const float* weight,
    device const float* anchor,
    device const float* scale_lut,
    device const char* first_codebooks,
    device const char* second_codebooks,
    device int* out_state,
    device int* out_first,
    device int* out_second,
    device float* out_group_error,
    constant uint& groups_per_batch,
    constant uint& groups_per_row,
    uint tid [[thread_index_in_threadgroup]],
    uint flat_group [[threadgroup_position_in_grid]]) {
  threadgroup float candidates[4 * 3 * 16];
  const uint batch = flat_group / groups_per_batch;
  const uint group = flat_group - batch * groups_per_batch;
  const uint row = group / groups_per_row;
  const uint vector = tid >> 5;
  const uint lane = tid & 31;
  const uint value_base = flat_group * 24;
  const uint table_base = batch * 4 * 8 * 4;
  if (lane < 16) {
    const bool second = lane >= 8;
    const uint entry = lane & 7;
    device const char* codebook = second ? second_codebooks : first_codebooks;
    const uint position_base = value_base + vector * 8 + (second ? 4 : 0);
    for (uint state = 0; state < 4; ++state) {
      float signal = 0.0f;
      float dot = 0.0f;
      float norm = 0.0f;
      for (uint coordinate = 0; coordinate < 4; ++coordinate) {
        const float source = value[position_base + coordinate];
        const float objective = weight[position_base + coordinate];
        const float code = float(codebook[
            table_base + (state * 8 + entry) * 4 + coordinate]);
        signal = fma(objective * source, source, signal);
        dot = fma(objective * source, code, dot);
        norm = fma(objective * code, code, norm);
      }
      const uint rows = groups_per_batch / groups_per_row;
      const float scale = anchor[batch * rows + row]
                        * scale_lut[batch * 4 + state];
      candidates[(state * 3 + vector) * 16 + lane] =
          fma(scale * scale, norm, fma(-2.0f * scale, dot, signal));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float best_group_error = INFINITY;
    uint best_state = 0;
    uint best_first[3] = {0, 0, 0};
    uint best_second[3] = {0, 0, 0};
    for (uint state = 0; state < 4; ++state) {
      float group_error = 0.0f;
      uint state_first[3] = {0, 0, 0};
      uint state_second[3] = {0, 0, 0};
      for (uint vector_id = 0; vector_id < 3; ++vector_id) {
        float first_error = INFINITY;
        float second_error = INFINITY;
        for (uint entry = 0; entry < 8; ++entry) {
          const float candidate_first =
              candidates[(state * 3 + vector_id) * 16 + entry];
          if (better(candidate_first, entry, first_error, state_first[vector_id])) {
            first_error = candidate_first;
            state_first[vector_id] = entry;
          }
          const float candidate_second =
              candidates[(state * 3 + vector_id) * 16 + 8 + entry];
          if (better(candidate_second, entry, second_error, state_second[vector_id])) {
            second_error = candidate_second;
            state_second[vector_id] = entry;
          }
        }
        group_error += first_error + second_error;
      }
      if (better(group_error, state, best_group_error, best_state)) {
        best_group_error = group_error;
        best_state = state;
        for (uint vector_id = 0; vector_id < 3; ++vector_id) {
          best_first[vector_id] = state_first[vector_id];
          best_second[vector_id] = state_second[vector_id];
        }
      }
    }
    out_state[flat_group] = int(best_state);
    out_group_error[flat_group] = best_group_error;
    for (uint vector_id = 0; vector_id < 3; ++vector_id) {
      out_first[flat_group * 3 + vector_id] = int(best_first[vector_id]);
      out_second[flat_group * 3 + vector_id] = int(best_second[vector_id]);
    }
  }
}
"""


@lru_cache(maxsize=1)
def _library():
    if not torch.backends.mps.is_available():
        raise RuntimeError("NPQ Metal assignment requires MPS")
    return torch.mps.compile_shader(_SOURCE)


def _assign(
    kernel_name: str,
    value: torch.Tensor,
    objective_weight: torch.Tensor,
    anchor: torch.Tensor,
    scale_lut: torch.Tensor,
    first_codebooks: torch.Tensor,
    second_codebooks: torch.Tensor,
    valid_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if value.device.type != "mps" or value.dtype != torch.float32:
        raise TypeError("NPQ Metal assignment expects MPS float32 values")
    rows, padded_width = map(int, value.shape)
    if not 0 < int(valid_width) <= padded_width or padded_width % 24:
        raise ValueError("invalid NPQ Metal assignment width")
    groups_per_row = padded_width // 24
    state = torch.empty((rows, groups_per_row), device="mps", dtype=torch.uint8)
    first = torch.empty((rows, groups_per_row, 3), device="mps", dtype=torch.uint8)
    second = torch.empty_like(first)
    group_error = torch.empty((rows, groups_per_row), device="mps", dtype=torch.float32)
    kernel = getattr(_library(), kernel_name)
    groups = rows * groups_per_row
    kernel(
        value.contiguous(),
        objective_weight.contiguous(),
        anchor.contiguous(),
        scale_lut.contiguous(),
        first_codebooks.contiguous(),
        second_codebooks.contiguous(),
        state,
        first,
        second,
        group_error,
        padded_width,
        int(valid_width),
        groups_per_row,
        threads=groups * 96,
        group_size=96,
    )
    return state, first, second, group_error


def npq0_s_assign(
    value: torch.Tensor,
    objective_weight: torch.Tensor,
    anchor: torch.Tensor,
    scale_lut: torch.Tensor,
    first_codebooks: torch.Tensor,
    second_codebooks: torch.Tensor,
    valid_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign NPQ0-S group states and the two 3-bit subvector indices."""

    if tuple(first_codebooks.shape) != (4, 8, 4):
        raise ValueError("NPQ0-S first codebook must have shape [4,8,4]")
    if tuple(second_codebooks.shape) != (4, 8, 4):
        raise ValueError("NPQ0-S second codebook must have shape [4,8,4]")
    return _assign(
        "npq0_s_assign", value, objective_weight, anchor, scale_lut,
        first_codebooks, second_codebooks, valid_width
    )


def npq0_l_assign(
    value: torch.Tensor,
    objective_weight: torch.Tensor,
    anchor: torch.Tensor,
    scale_lut: torch.Tensor,
    first_codebooks: torch.Tensor,
    second_codebooks: torch.Tensor,
    valid_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign NPQ0-L group states and its 3-bit/4-bit subvector indices."""

    if tuple(first_codebooks.shape) != (8, 8, 4):
        raise ValueError("NPQ0-L first codebook must have shape [8,8,4]")
    if tuple(second_codebooks.shape) != (8, 16, 4):
        raise ValueError("NPQ0-L second codebook must have shape [8,16,4]")
    return _assign(
        "npq0_l_assign", value, objective_weight, anchor, scale_lut,
        first_codebooks, second_codebooks, valid_width
    )


def npq0_s_batched_assign(
    value: torch.Tensor,
    objective_weight: torch.Tensor,
    anchor: torch.Tensor,
    scale_lut: torch.Tensor,
    first_codebooks: torch.Tensor,
    second_codebooks: torch.Tensor,
    *,
    rows: int,
    groups_per_row: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign one independently trained NPQ0-S table per leading batch."""

    if value.device.type != "mps" or value.dtype != torch.float32:
        raise TypeError("batched NPQ0-S Metal assignment expects MPS float32")
    batches, groups_per_batch, width = map(int, value.shape)
    if width != 24 or groups_per_batch != int(rows) * int(groups_per_row):
        raise ValueError("invalid batched NPQ0-S group geometry")
    if tuple(first_codebooks.shape) != (batches, 4, 8, 4):
        raise ValueError("batched NPQ0-S first tables must be [batch,4,8,4]")
    if tuple(second_codebooks.shape) != (batches, 4, 8, 4):
        raise ValueError("batched NPQ0-S second tables must be [batch,4,8,4]")
    total_groups = batches * groups_per_batch
    state = torch.empty(
        (batches, groups_per_batch), device="mps", dtype=torch.int32
    )
    first = torch.empty(
        (batches, groups_per_batch, 3), device="mps", dtype=torch.int32
    )
    second = torch.empty_like(first)
    error = torch.empty(
        (batches, groups_per_batch), device="mps", dtype=torch.float32
    )
    _library().npq0_s_batched_assign(
        value.contiguous(),
        objective_weight.contiguous(),
        anchor.contiguous(),
        scale_lut.contiguous(),
        first_codebooks.contiguous(),
        second_codebooks.contiguous(),
        state,
        first,
        second,
        error,
        groups_per_batch,
        int(groups_per_row),
        threads=total_groups * 96,
        group_size=96,
    )
    return state.to(torch.int64), first.to(torch.int64), second.to(torch.int64), error


__all__ = ["npq0_l_assign", "npq0_s_assign", "npq0_s_batched_assign"]
