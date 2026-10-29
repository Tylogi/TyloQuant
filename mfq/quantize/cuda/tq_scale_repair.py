"""Cheap row-scale proposals; callers accept only measured SSE improvements."""
from functools import lru_cache

import torch

from mfq.formats.tq import fixed_codebook

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


@lru_cache(None)
def _bounds():
    book = fixed_codebook()
    return float(book.min()), float(book.max())


@torch.inference_mode()
def _torch_proposals(w, h, columns, anchors, loss):
    low, high = _bounds()
    a = anchors.float()
    magnitude = w.abs()
    edge = columns.float() * torch.where(w >= 0, high, -low)
    clipped = (magnitude - a[:, None] * edge).clamp_min(0)
    beta = (loss - (clipped.square() * h).sum(-1)).clamp_min(0) / a.square().clamp_min(1e-30)
    candidate = a.clone()
    for _ in range(4):
        active = magnitude > candidate[:, None] * edge
        numerator = torch.where(active, h * magnitude * edge, 0).sum(-1)
        denominator = beta + torch.where(active, h * edge.square(), 0).sum(-1)
        candidate = torch.maximum(candidate, numerator / denominator.clamp_min(1e-30))
    candidate = candidate.half().float()
    clipped = (magnitude - candidate[:, None] * edge).clamp_min(0)
    predicted = (clipped.square() * h).sum(-1) + beta * candidate.square()
    return torch.where((predicted < loss) & (candidate > a) & (a > 0), candidate, a).half()


if triton is not None:
    @triton.jit
    def _propose(W, H, C, A, L, P, K: tl.constexpr,
                 LOW: tl.constexpr, HIGH: tl.constexpr, B: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, B)
        w = tl.load(W + row * K + col, col < K, 0).to(tl.float32)
        h = tl.load(H + col, col < K, 0).to(tl.float32)
        c = tl.load(C + col, col < K, 0).to(tl.float32)
        a = tl.load(A + row).to(tl.float32)
        loss = tl.load(L + row).to(tl.float32)
        magnitude = tl.abs(w)
        edge = c * tl.where(w >= 0, HIGH, -LOW)
        clipped = tl.maximum(magnitude - a * edge, 0.)
        beta = tl.maximum(loss - tl.sum(h * clipped * clipped, 0), 0.) / tl.maximum(a * a, 1.e-30)
        candidate = a
        for _ in range(4):
            active = magnitude > candidate * edge
            numerator = tl.sum(tl.where(active, h * magnitude * edge, 0.), 0)
            denominator = beta + tl.sum(tl.where(active, h * edge * edge, 0.), 0)
            candidate = tl.maximum(candidate, numerator / tl.maximum(denominator, 1.e-30))
        candidate = candidate.to(tl.float16).to(tl.float32)
        clipped = tl.maximum(magnitude - candidate * edge, 0.)
        predicted = tl.sum(h * clipped * clipped, 0) + beta * candidate * candidate
        candidate = tl.where((predicted < loss) & (candidate > a) & (a > 0), candidate, a)
        tl.store(P + row, candidate)


@torch.inference_mode()
def propose_scales(w, h, columns, anchors, loss):
    """Minimize a clipping-plus-quadratic surrogate in four active-set steps.

    The quadratic term matches the measured non-clipping error at the old
    anchor. This only proposes scales: the encoder must compare actual FP16
    weighted SSE before replacing any row. Triton is an optional accelerator.
    """
    if triton is None or not w.is_cuda or not all(
        t.is_contiguous() for t in (w, h, columns, anchors, loss)
    ):
        return _torch_proposals(w, h, columns, anchors, loss)
    with torch.cuda.device(w.device):
        rows, width = w.shape
        result = torch.empty_like(anchors)
        low, high = _bounds()
        _propose[(rows,)](w, h, columns, anchors, loss, result, width, low, high,
                         triton.next_power_of_2(width), num_warps=8 if width > 4096 else 4)
        return result
