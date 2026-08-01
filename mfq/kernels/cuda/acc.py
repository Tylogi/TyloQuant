"""Residual add (thin glue; kernel in acc.cu)."""

from __future__ import annotations

import torch

from mfq.kernels.cuda._ext import ext


def acc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a + b`` residual add (ggml acc). Preserves f16/f32 dtype."""

    if a.dtype != b.dtype:
        dtype = torch.promote_types(a.dtype, b.dtype)
        if dtype not in (torch.float16, torch.float32):
            dtype = torch.float32
        a = a.to(dtype)
        b = b.to(dtype)
    elif a.dtype not in (torch.float16, torch.float32):
        a = a.to(torch.float32)
        b = b.to(torch.float32)
    return ext().acc_cuda(a.contiguous(), b.contiguous())
