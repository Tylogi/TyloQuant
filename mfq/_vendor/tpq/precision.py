"""TPQ precision-policy layer: all framework compute paths obtain their compute dtype here.

Motivation: VQ/int4 quantization error (relative ~0.3-0.4) is over two orders of magnitude larger than
half-precision GEMM rounding error (fp16 ~1e-3, bf16 ~8e-3). GEMM dot products can therefore use
half-precision tensor cores with negligible impact on output distributions. The acceptance gate requires
token-for-token agreement between greedy and speculative dspark_check outputs plus a KL difference below 0.01.

Automatic policy, adapting to hardware capabilities without code changes when switching devices:
  - sm_8.x+ (Ampere and newer: 3090/4090/A100/H100) -> bf16 for good dynamic range without overflow;
  - sm_7.x (Turing: 2080/T4, without bf16 hardware) -> fp16 tensor cores at about 2x fp32 GEMM speed;
  - CPU / other -> fp32 because half precision does not accelerate CPU and some operators do not support it.
The TPQ_COMPUTE_DTYPE=fp32|fp16|bf16 environment variable can force an override for debugging/comparison.
"""

from __future__ import annotations

import os

import torch

_AUTO_CACHE: dict[str, torch.dtype] = {}


def compute_dtype(device=None) -> torch.dtype:
    """Return the GEMM compute dtype for the current device; see the policy table in the module docstring."""
    ov = os.environ.get("TPQ_COMPUTE_DTYPE", "auto").strip().lower()
    if ov in ("fp32", "float32", "f32"):
        return torch.float32
    if ov in ("fp16", "float16", "f16", "half"):
        return torch.float16
    if ov in ("bf16", "bfloat16"):
        return torch.bfloat16
    dev = torch.device(device) if device is not None else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        return torch.float32
    key = str(dev)
    dt = _AUTO_CACHE.get(key)
    if dt is None:
        try:
            major, _minor = torch.cuda.get_device_capability(dev)
        except Exception:
            major = 0
        dt = torch.bfloat16 if major >= 8 else \
            (torch.float16 if major == 7 else torch.float32)
        _AUTO_CACHE[key] = dt
    return dt
