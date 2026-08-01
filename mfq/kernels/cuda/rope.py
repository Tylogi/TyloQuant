"""Rotary Position Embedding (thin glue; kernel in rope.cu)."""

from __future__ import annotations

import os

import torch

from mfq.kernels.cuda._ext import ext

_ROPE_TABLES: dict[tuple[int, float, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _device_key(device: torch.device) -> int:
    if device.type != "cuda":
        return -1
    return torch.cuda.current_device() if device.index is None else int(device.index)


def _rope_table(device: torch.device, base: float, rotary_dim: int, table_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    table_len = int(table_len)
    key = (_device_key(device), float(base), int(rotary_dim), table_len)
    hit = _ROPE_TABLES.get(key)
    if hit is not None:
        return hit
    half = int(rotary_dim) // 2
    pos = torch.arange(table_len, device=device, dtype=torch.float32)
    freq = base ** (-torch.arange(0, int(rotary_dim), 2, device=device, dtype=torch.float32) / float(rotary_dim))
    ang = pos[:, None] * freq[None, :]
    cos = torch.cos(ang).contiguous()
    sin = torch.sin(ang).contiguous()
    assert cos.shape == (table_len, half)
    _ROPE_TABLES[key] = (cos, sin)
    return cos, sin


def rope(
    x: torch.Tensor,
    pos_ids: torch.Tensor,
    base: float = 1_000_000.0,
    rotary_dim: int | None = None,
    sections: tuple[int, int, int] | None = None,
    table_len: int | None = None,
) -> torch.Tensor:
    """Rotate-half RoPE. Supports partial RoPE and MRoPE ``pos_ids [3,T]``."""

    xf = x.contiguous().to(torch.float32)
    rotary = int(xf.shape[-1] if rotary_dim is None else rotary_dim)
    sec = torch.empty((0,), dtype=torch.int64)
    if sections is not None:
        sec = torch.tensor(tuple(int(v) for v in sections), dtype=torch.int64)
    if table_len is not None or os.environ.get("MFQ_ROPE_TABLE", "1") != "0":
        if table_len is None:
            # Non-runtime callers get a small table sized from current positions.
            table_len = max(16, int(pos_ids.max().item()) + 1)
        pos_i = pos_ids.contiguous().to(device=x.device, dtype=torch.int64)
        cos, sin = _rope_table(x.device, float(base), rotary, int(table_len))
        return ext().rope_table_cuda(xf, pos_i, cos, sin, rotary, sec)
    pos = pos_ids.contiguous().to(device=x.device, dtype=torch.float32)
    if rotary_dim is None and sections is None:
        return ext().rope_cuda(xf, pos, float(base))
    return ext().rope_ext_cuda(xf, pos, float(base), rotary, sec)
