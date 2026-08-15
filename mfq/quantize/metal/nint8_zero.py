"""Metal encoder for the GGML-compatible NINT8-0 block format."""

from __future__ import annotations

from functools import lru_cache

import torch

from mfq.formats.nint8_zero import Nint8ZeroTensor

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void nint8_zero(
    device const float* weight,
    device half* out_scale,
    device char* out_q,
    uint tid [[thread_index_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
  threadgroup float maximum_parts[32];
  const uint offset = block * 32;
  const float source = weight[offset + tid];
  maximum_parts[tid] = abs(source);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 16; stride > 0; stride >>= 1) {
    if (tid < stride) {
      maximum_parts[tid] = max(
          maximum_parts[tid], maximum_parts[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float scale = maximum_parts[0] / 127.0f;
  if (tid == 0) {
    out_scale[block] = half(scale);
  }
  int quantized = 0;
  if (scale > 0.0f) {
    const float normalized = source / scale;
    const float rounded = copysign(floor(abs(normalized) + 0.5f), normalized);
    quantized = clamp(int(rounded), -127, 127);
  }
  out_q[offset + tid] = char(quantized);
}
"""


@lru_cache(maxsize=1)
def _library():
    if not torch.backends.mps.is_available():
        raise RuntimeError("NINT8-0 Metal quantization requires MPS")
    return torch.mps.compile_shader(_SOURCE)


def quantize(
    weight: torch.Tensor,
    *,
    axis: int = 0,
) -> Nint8ZeroTensor:
    """Encode one tensor using the deterministic Q8_0 rounding rule."""

    if weight.device.type != "mps":
        raise TypeError("NINT8-0 Metal quantization expects an MPS tensor")
    if weight.ndim == 0 or not 0 <= int(axis) < weight.ndim:
        raise ValueError("NINT8-0 axis is outside the tensor rank")
    value = weight.to(dtype=torch.float32)
    moved = value.movedim(int(axis), 0).contiguous()
    shape = tuple(map(int, value.shape))
    out = int(moved.shape[0])
    neuron_len = moved.numel() // out
    if neuron_len <= 0 or neuron_len % 32:
        raise ValueError("NINT8-0 neuron_len must be a positive multiple of 32")
    blocks = out * neuron_len // 32
    scale = torch.empty(blocks, device="mps", dtype=torch.float16)
    q = torch.empty(out * neuron_len, device="mps", dtype=torch.int8)
    _library().nint8_zero(
        moved.reshape(out, neuron_len),
        scale,
        q,
        threads=blocks * 32,
        group_size=32,
    )
    return Nint8ZeroTensor(
        shape=shape,
        axis=int(axis),
        scale=scale.reshape(out, neuron_len // 32).cpu().numpy(),
        q=q.reshape(out, neuron_len // 32, 32).cpu().numpy(),
        neuron_len=neuron_len,
    )


__all__ = ["quantize"]
