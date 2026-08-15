"""Metal nearest-record search for NEPQ-A additive residual streams."""

from __future__ import annotations

from functools import lru_cache

import torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

inline bool better(float gain, uint index, float best_gain, uint best_index) {
  return gain > best_gain || (gain == best_gain && index < best_index);
}

kernel void nepq_a_best_records(
    device const float* blocks,
    device const float* dictionary,
    device const float* dictionary_norm,
    device const int* valid_vectors,
    device int* out_records,
    device float* out_gains,
    constant uint& block_vectors,
    constant uint& position_bits,
    uint tid [[thread_index_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
  threadgroup float best_gain_parts[256];
  threadgroup uint best_index_parts[256];
  const uint valid = uint(valid_vectors[block]);
  const uint candidates = valid * 1024;
  float best_gain = -INFINITY;
  uint best_index = 0;
  for (uint candidate = tid; candidate < candidates; candidate += 256) {
    const uint position = candidate >> 10;
    const uint dictionary_id = candidate & 1023;
    float dot = 0.0f;
    const uint block_offset = (block * block_vectors + position) * 8;
    const uint dictionary_offset = dictionary_id * 8;
    for (uint coordinate = 0; coordinate < 8; ++coordinate) {
      dot = fma(
          blocks[block_offset + coordinate],
          dictionary[dictionary_offset + coordinate],
          dot);
    }
    const float gain = 2.0f * dot - dictionary_norm[dictionary_id];
    if (better(gain, candidate, best_gain, best_index)) {
      best_gain = gain;
      best_index = candidate;
    }
  }
  best_gain_parts[tid] = best_gain;
  best_index_parts[tid] = best_index;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride && better(
            best_gain_parts[tid + stride],
            best_index_parts[tid + stride],
            best_gain_parts[tid],
            best_index_parts[tid])) {
      best_gain_parts[tid] = best_gain_parts[tid + stride];
      best_index_parts[tid] = best_index_parts[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    const uint position = best_index_parts[0] >> 10;
    const uint dictionary_id = best_index_parts[0] & 1023;
    out_records[block] = int(position | (dictionary_id << position_bits));
    out_gains[block] = best_gain_parts[0];
  }
}
"""


@lru_cache(maxsize=1)
def _library():
    if not torch.backends.mps.is_available():
        raise RuntimeError("NEPQ-A Metal record search requires MPS")
    return torch.mps.compile_shader(_SOURCE)


def best_records(
    blocks: torch.Tensor,
    dictionary: torch.Tensor,
    dictionary_norm: torch.Tensor,
    valid_vectors: torch.Tensor,
    position_bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the maximum-gain residual record for every block."""

    if blocks.device.type != "mps" or blocks.dtype != torch.float32:
        raise TypeError("NEPQ-A Metal search expects MPS float32 blocks")
    total, block_vectors, vector_size = map(int, blocks.shape)
    if vector_size != 8 or tuple(dictionary.shape) != (1024, 8):
        raise ValueError("NEPQ-A Metal search requires 8-D 1024-entry tables")
    if not 0 < int(position_bits) < 16:
        raise ValueError("invalid NEPQ-A residual position width")
    records_i32 = torch.empty(total, device="mps", dtype=torch.int32)
    gains = torch.empty(total, device="mps", dtype=torch.float32)
    _library().nepq_a_best_records(
        blocks.contiguous(),
        dictionary.contiguous(),
        dictionary_norm.contiguous(),
        valid_vectors.to(torch.int32).contiguous(),
        records_i32,
        gains,
        block_vectors,
        int(position_bits),
        threads=total * 256,
        group_size=256,
    )
    return records_i32.to(torch.int64), gains


__all__ = ["best_records"]
