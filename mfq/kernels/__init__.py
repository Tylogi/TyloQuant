"""Hardware-backend kernels.

Mirrors the ggml convention of one file per op. Native CUDA implementations all live in
:mod:`mfq.kernels.cuda`, compiled into a single extension ``mfq_cuda`` (the llama.cpp analog
of one libggml-cuda):

- ``cuda/norm.cu``            - rms_norm / l2_norm
- ``cuda/rope.cu``            - RoPE
- ``cuda/attention.cu``       - full / GQA attention (online softmax; fattn.cu)
- ``cuda/acc.cu``             - residual add
- ``cuda/gated_delta_net.cu`` - GDN linear-attention recurrence (most Qwen3.5 layers)
- ``cuda/kv_cache.cu``        - KV cache write
- ``cuda/embedding.cu``       - token embedding lookup / selected-row NINT dequant
- ``cuda/sampling.cu``        - logits sampling
- ``cuda/nint_matmul.cu``     - NINT INT-fused-GEMM (weight path, dequant-during-GEMM)

This level keeps pure Python/numpy reference implementations for correctness checks and for
testing without CUDA:

- ``gated_delta_net`` - GDN recurrence reference (vs ``cuda/gated_delta_net.cu``)
- ``torch_backend``   - NINT dequant + decomposed matmul (materialize path, vs fused kernel)

The Apple-silicon backend in ``metal/`` provides MLX custom Metal kernels for
packed NINT/NVQ/NPQ/NEPQ GEMV, MMQ, and GEMM, packed embedding, GDN/SSM, plus reusable
norm, RoPE, activation, and residual operations.
"""
