"""Quantization execution logic.

This subpackage converts a full-precision tensor to an MFQ-specified precision:

- ``nint_quant`` -- neuron-anchored INT tensor-level quantization (the main weight path).
- ``nint_quant_torch`` -- CUDA/Metal NINT tensor-level quantization (the main large-model conversion path).
- ``nvq_quant`` -- joint search over E8/D4 codewords and neuron/sub-group scales.
- ``nvq_quant_torch`` -- offline CUDA/Metal quantization and native assignment-kernel dispatch for NVQ1-L/NVQ2/NVQ3.
- ``nvq_jsc`` -- calibration-free NVQ2J joint scale/codebook-state training and streamed fixed-table quantization.
- ``npq0_s`` -- NPQ0-S tensor-wise PQ3+3 training and fixed-table quantization.
- ``nvq1_l_quant`` -- joint search over 8-D ternary codewords, deltas, and neuron/sub-group scales.
- ``nvq_tensor_codebook`` -- per-tensor codebook training and held-out selection for NVQ1-L/NVQ2/NVQ3.
- ``imatrix`` -- reads llama.cpp GGUF/legacy activation-importance matrices and binds them to tensors.
- ``nvq1_s_quant`` / ``nvq1_s_quant_torch`` -- CPU and native Metal NVQ1-S 512-entry ternary solvers.
- ``nvq1_s_codebook`` -- weighted NVQ1-S codebook training and unique ternary projection.
- ``npq0_l`` -- NPQ0-L state-conditioned PQ3+4 codebook training and fixed-table quantization.
- ``nepq`` -- cross-expert shared codebook pools, 96-weight bank selection, and four fixed-pool NEPQ quantizers.
- ``tpq`` -- native Metal/CPU TPQ product-codebook training, assignment, and TPQ-I4 encoding.
- ``ternary_quant`` -- joint search over scalar ternary values and neuron/sub-group scales.
- ``nvq_codebook`` -- NVQ2 E8 codebook trainer for a model, tensor family, or individual tensor.
- ``sensitivity`` -- per-tensor sensitivity analysis used for precision allocation.
- ``rotate`` -- Hadamard rotation to improve quantizability of ultra-low-precision layers and critical tensors.
"""
