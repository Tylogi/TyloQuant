"""MFQ 原生格式定义。

- ``nint``   — Neuron-anchored INT 编解码（MFQ 核心权重格式：neuron 锚定两级 scale）。
- ``nvq``    — Neuron-anchored E8/D4/JSC lattice VQ（2/3-bit 超低精度格式）。
- ``nvq1_l``   — Neuron-anchored 8-D ternary lattice VQ（1.x-bit 格式）。
- ``nvq1_s``   — uint32/gs24 的 512-entry ternary VQ（约 1.34 bpw）。
- ``npq0_l`` — gs24、PQ3+4 与逐 neuron 锚点的约 1.00 bpw 格式。
- ``npq0_s`` — gs24、state-conditioned PQ3+3 的约 0.84 bpw 格式。
- ``nepq`` — 跨专家共享表池，每 4 个 gs24 组保存一个 uint8 bank ID。
- ``ternary`` — 5 trits/byte 的 neuron-anchored scalar ternary 对照格式。
- ``scheme`` — 精度方案，描述每个 tensor 的权重/激活精度（如 MFQ-W4.51）。
- ``header`` — MFQ 文件头与张量级元数据。
- ``io``     — MFQ 文件的序列化 / 反序列化。
"""
