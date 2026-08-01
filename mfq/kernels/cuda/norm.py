"""RMSNorm / L2 Norm (thin glue; kernel in norm.cu)."""

from __future__ import annotations

import torch

from mfq.kernels.cuda._ext import ext


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``x * rsqrt(mean(x^2)+eps) * weight`` over the last dim (ggml rms_norm). fp32/cuda."""

    return ext().rms_norm_cuda(
        x.contiguous().to(torch.float32), weight.contiguous().to(torch.float32), float(eps)
    )


def l2_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Row-wise L2 normalize the last dim: ``x / max(||x||_2, eps)`` (ggml l2_norm)."""

    return ext().l2_norm_cuda(x.contiguous().to(torch.float32), float(eps))
