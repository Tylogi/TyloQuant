"""SSM convolution helpers."""

from __future__ import annotations

import torch

from mfq.kernels.cuda._ext import ext


def ssm_conv_silu(
    conv_input: torch.Tensor,
    weight: torch.Tensor,
    n_tokens: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused causal depthwise SSM conv + SiLU on CUDA."""

    if bias is None:
        bias = torch.empty(0, device=conv_input.device, dtype=torch.float32)
    return ext().ssm_conv_silu_cuda(
        conv_input.contiguous().to(torch.float32),
        weight.contiguous().to(torch.float32),
        bias.contiguous().to(device=conv_input.device, dtype=torch.float32),
        int(n_tokens),
    )
