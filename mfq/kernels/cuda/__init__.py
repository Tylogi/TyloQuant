"""CUDA 后端 kernel（NVIDIA GPU）。

基本路径由 :mod:`mfq.kernels.torch_backend` 覆盖（torch + CUDA：NINT 两级 scale
反量化 + llama.cpp 式分解 matmul）。后续可经 ``torch.utils.cpp_extension`` 注入原生
INT-fused-GEMM（dequant-during-GEMM，不 materialize Wq），上层零改动。
"""

from __future__ import annotations
