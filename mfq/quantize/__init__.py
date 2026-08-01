"""量化执行逻辑。

本子包负责「把一个全精度 tensor 变成 MFQ 指定精度」：

- ``nint_quant``  — neuron-anchored INT 张量级量化（核心权重路径）。
- ``nint_quant_torch`` — torch/CUDA 版 NINT 张量级量化（大模型转换主路径）。
- ``nvq_quant``   — E8/D4 码字与 neuron/sub-group scale 联合搜索。
- ``nvq_quant_torch`` — NVQ1-L/NVQ2/NVQ3 CUDA 离线量化与原生 assignment kernel 调度。
- ``nvq_jsc`` — 默认无校准 NVQ2J 联合 scale/codebook-state 训练与流式固定表量化。
- ``npq0_s`` — NPQ0-S tensor-wise PQ3+3 训练与固定表量化。
- ``nvq1_l_quant`` — 8-D ternary 码字、delta 与 neuron/sub-group scale 联合搜索。
- ``nvq_tensor_codebook`` — NVQ1-L/NVQ2/NVQ3 逐 tensor 码本训练与 held-out 选择。
- ``imatrix`` — llama.cpp GGUF/legacy 激活重要性矩阵读取与 tensor 绑定。
- ``nvq1_s_quant`` — NVQ1-S 的 512-entry ternary VQ 与固定 32-bit group 求解。
- ``nvq1_s_codebook`` — NVQ1-S 码本的加权训练与唯一 ternary 投影。
- ``npq0_l`` — NPQ0-L 的 state-conditioned PQ3+4 码本训练与固定表量化。
- ``nepq`` — 跨专家共享码本池、96-weight bank 选择与四种 NEPQ 固定池量化。
- ``ternary_quant`` — scalar ternary 与 neuron/sub-group scale 联合搜索。
- ``nvq_codebook`` — 模型、tensor 家族或单 tensor 的 NVQ2 E8 码本训练器。
- ``sensitivity`` — 逐 tensor 灵敏度分析，为精度分配提供输入。
- ``rotate``      — Hadamard rotation，提升超低精度层与关键 tensor 的可量化性。
"""
