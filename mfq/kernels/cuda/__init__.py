"""CUDA backend kernels for NVIDIA GPUs.

The basic path is provided by :mod:`mfq.kernels.torch_backend` (torch plus CUDA: NINT two-level-scale
dequantization and llama.cpp-style decomposed matmul). A native INT-fused-GEMM can later be injected through
``torch.utils.cpp_extension`` to dequantize during GEMM without materializing Wq, requiring no upper-layer changes.
"""

from __future__ import annotations
