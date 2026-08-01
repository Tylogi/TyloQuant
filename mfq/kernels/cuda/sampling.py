"""GPU sampling from logits."""

from __future__ import annotations

import torch

from mfq.kernels.cuda._ext import ext


def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    """Argmax over the last dim, returning int64 token ids."""

    flat = logits.contiguous().view(-1, logits.shape[-1])
    out = ext().sample_greedy_cuda(flat)
    return out.view(logits.shape[:-1])


def sample(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    random: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample token ids from logits on GPU.

    ``temperature <= 0`` or ``top_k == 1`` uses greedy. ``top_p`` is applied within
    ``top_k`` when ``top_k > 0``.
    """

    flat = logits.contiguous().view(-1, logits.shape[-1])
    if temperature <= 0.0 or top_k == 1:
        out = ext().sample_greedy_cuda(flat)
        return out.view(logits.shape[:-1])
    if random is None:
        random = torch.rand((flat.shape[0],), device=flat.device, dtype=torch.float32)
    else:
        random = random.contiguous().to(device=flat.device, dtype=torch.float32).view(-1)
    if top_k > 0:
        out = ext().sample_top_k_top_p_cuda(flat, random, float(temperature), int(top_k), float(top_p))
    else:
        if top_p < 1.0:
            raise ValueError("top_p requires top_k > 0 in the CUDA sampler")
        out = ext().sample_softmax_cuda(flat, random, float(temperature))
    return out.view(logits.shape[:-1])
