"""Offline quantization backend selection.

The packed MFQ writers are shared across platforms.  CUDA and Apple Metal
select the same Torch tensor solvers while retaining backend-specific fused
operations where they exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


ACCELERATOR_BACKENDS = frozenset({"cuda", "metal"})
QUANT_BACKENDS = ("auto", "cuda", "metal", "cpu")


@dataclass(frozen=True)
class QuantBackend:
    name: str
    device: str

    @property
    def accelerated(self) -> bool:
        return self.name in ACCELERATOR_BACKENDS


def metal_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def resolve_quant_backend(requested: str, device: str = "") -> QuantBackend:
    name = str(requested).lower()
    if name not in QUANT_BACKENDS:
        raise ValueError(f"unsupported quant backend: {requested}")
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif metal_available():
            name = "metal"
        else:
            name = "cpu"
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA quant backend requested but CUDA is unavailable")
        selected = device if str(device).startswith("cuda") else "cuda"
    elif name == "metal":
        if not metal_available():
            raise RuntimeError("Metal quant backend requested but PyTorch MPS is unavailable")
        selected = "mps"
    else:
        selected = "cpu"
    return QuantBackend(name=name, device=selected)


def resolve_row_chunk(requested: int, backend: str) -> int:
    value = int(requested)
    if value < 0:
        raise ValueError("row chunk must be non-negative")
    if value:
        return value
    return 8192 if backend == "metal" else 1024


__all__ = [
    "ACCELERATOR_BACKENDS",
    "QUANT_BACKENDS",
    "QuantBackend",
    "metal_available",
    "resolve_quant_backend",
    "resolve_row_chunk",
]
