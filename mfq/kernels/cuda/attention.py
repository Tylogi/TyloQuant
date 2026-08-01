"""Full / Grouped-Query Attention (thin glue; kernel in attention.cu)."""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from mfq.kernels.cuda._ext import ext


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    scale: float | None = None,
) -> torch.Tensor:
    """``softmax((q k^T)*scale) v`` via online softmax.

    q: ``[B, H_q, T, D]``; k, v: ``[B, H_kv, Tk, D]`` (H_q % H_kv == 0, GQA).
    causal: query tq sees key s iff s <= tq + Tk - T (matches torch SDPA).
    scale: default 1/sqrt(D).
    """

    if k.dtype == torch.float16 and v.dtype == torch.float16:
        dtype = torch.float16
    elif q.dtype == torch.float16 and k.dtype == torch.float16 and v.dtype == torch.float16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    qf = q.contiguous().to(dtype)
    kf = k.contiguous().to(dtype)
    vf = v.contiguous().to(dtype)
    if scale is None:
        scale = 1.0 / math.sqrt(qf.size(-1))
    if qf.size(2) == kf.size(2) and os.environ.get("MFQ_ATTENTION_SDPA", "1") != "0":
        rep = qf.size(1) // kf.size(1)
        if rep != 1:
            kf = kf.repeat_interleave(rep, dim=1)
            vf = vf.repeat_interleave(rep, dim=1)
        return F.scaled_dot_product_attention(qf, kf, vf, is_causal=bool(causal), scale=float(scale))
    return ext().attention_cuda(qf, kf, vf, float(scale), bool(causal))


def sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Causal attention over the last ``window`` keys for each query."""

    if window <= 0:
        raise ValueError("window must be positive")
    dtype = torch.float16 if q.dtype == k.dtype == v.dtype == torch.float16 else torch.float32
    qf = q.contiguous().to(dtype)
    kf = k.contiguous().to(dtype)
    vf = v.contiguous().to(dtype)
    if scale is None:
        scale = 1.0 / math.sqrt(qf.size(-1))
    return ext().attention_swa_cuda(qf, kf, vf, float(scale), int(window))


def sliding_window_attention_cached(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_len: torch.Tensor,
    window: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Attend with the newest queries over a circular SWA cache."""

    if window <= 0:
        raise ValueError("window must be positive")
    dtype = k_cache.dtype
    qf = q.contiguous().to(dtype)
    if scale is None:
        scale = 1.0 / math.sqrt(qf.size(-1))
    return ext().attention_cache_swa_cuda(
        qf,
        k_cache.contiguous(),
        v_cache.contiguous(),
        seq_len.contiguous().to(device=q.device, dtype=torch.int64),
        float(scale),
        int(window),
    )
