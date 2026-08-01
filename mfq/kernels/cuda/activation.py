"""Activation helpers."""

from __future__ import annotations

import torch

from mfq.kernels.cuda._ext import ext


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Return ``silu(gate) * up`` on CUDA for f16/f32 tensors."""

    return ext().silu_mul_cuda(gate.contiguous(), up.contiguous())


def gelu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Return ``gelu(gate, approximate='tanh') * up`` on CUDA."""

    return ext().gelu_mul_cuda(gate.contiguous(), up.contiguous())
