# DeepSeek-V4-Flash-0731：MFQ、TPQ 与 UD 结果

更新时间：2026-08-06

## 测试协议

- 源权重：官方 DeepSeek-V4-Flash-0731。
- Reference：官方 DeepSeek runtime 生成的 BF16 logits。
- 数据集：WikiText-2 `wiki.test.raw`，使用已校验的原始字节、CRLF 与 token 对齐。
- 每条序列：`ctx=512`；reference 使用 `batch=512`、`ubatch=512`、`n_seq=1`。
- 规模：573 个 chunk，每块打分 255 个 token，共 146,115 个 token。
- Mean KLD：正向 `KL(P_reference || P_model)`；same-top 为 reference 与被测模型 argmax 一致率。
- UD：llama.cpp 全驻留评测。
- TPQ：MFQ 内的 TPQ 全驻留路径。
- MFQ：streamed TP1 路径、`layer_group=4`、FP16 MMQ。表中 B4 表示一次并行处理 4 条互不共享注意力状态的 512-token 序列；单序列 context 仍为 512，总执行 batch 为 2048。
- 只列完整 573-chunk 结果；三块门、失败和中途结果均未计入。

## 完整结果

| 类型 | 模型 | 字节数 | 大小（GiB） | Mean KLD | same-top | 执行方式 |
|---|---|---:|---:|---:|---:|---|
| MFQ | DeepSeek-V4-Flash-0731-EW-V2-S（NINT8-0 Emb/Head，发布版） | 83,235,658,220 | 77.519 | **0.313576** | **82.2913%** | streamed TP1x6, B4 |
| MFQ | DeepSeek-V4-Flash-0731-EW-V2-S（BF16 Emb/Head） | 84,228,528,563 | 78.444 | 0.313559 | 82.2386% | streamed TP1x6, B4 |
| MFQ | DeepSeek-V4-Flash-0731-EW-V2-M | 94,496,321,567 | 88.007 | **0.244488** | **84.5300%** | streamed TP1x6, B4 |
| MFQ | DeepSeek-V4-Flash-0731-EW-V2-L | 105,234,452,170 | 98.007 | **0.201444** | **86.0753%** | streamed TP1x6, B4 |
| TPQ | DeepSeek-V4-Flash-0731-TPQ-S | 82,112,397,367 | 76.473 | 0.291115 | 83.0846% | full-resident TP1x6 |
| UD | UD-IQ1_S | 82,539,237,792 | 76.871 | 0.645514 | 73.0110% | llama.cpp |
| UD | UD-IQ1_M | 86,901,313,952 | 80.933 | 0.581024 | 74.6580% | llama.cpp |
| UD | UD-IQ2_XXS | 90,860,736,928 | 84.621 | 0.478268 | 76.9380% | llama.cpp |
| UD | UD-IQ2_M | 90,926,928,288 | 84.682 | 0.478002 | 76.8580% | llama.cpp |
| UD | UD-Q2_K_XL | 96,832,508,352 | 90.182 | 0.403276 | 78.7780% | llama.cpp |
| UD | UD-IQ3_XXS | 104,207,848,032 | 97.051 | 0.306343 | 82.0150% | llama.cpp |
| UD | UD-IQ3_S | 116,069,339,712 | 108.098 | 0.310893 | 81.6860% | llama.cpp |
| UD | UD-Q3_K_M | 128,078,484,032 | 119.282 | 0.215570 | 85.1060% | llama.cpp |
| UD | UD-Q3_K_XL | 128,206,729,792 | 119.402 | 0.215313 | 84.9650% | llama.cpp |
| UD | UD-IQ4_NL | 136,662,446,656 | 127.277 | 0.180695 | 86.0750% | llama.cpp |
| UD | UD-Q4_K_XL | 155,095,241,120 | 144.444 | 0.149590 | 87.4540% | llama.cpp |
| UD | UD-Q8_K_XL | 161,869,615,520 | 150.753 | 0.149420 | 87.5870% | llama.cpp |

## 近似同体积对位

| 对位 | 大小差（MFQ − UD） | KLD 差（MFQ − UD） | KLD 降幅 | same-top 差 |
|---|---:|---:|---:|---:|
| EW-V2-S 发布版 / UD-IQ1_S | +0.649 GiB | -0.331938 | **51.422%** | **+9.2803 pp** |
| EW-V2-M / UD-Q2_K_XL | -2.176 GiB | -0.158788 | **39.374%** | **+5.7520 pp** |
| EW-V2-L / UD-IQ3_XXS | +0.956 GiB | -0.104899 | **34.242%** | **+4.0603 pp** |

## B4 与 B1 数值检查

BF16 Emb/Head 的 EW-V2-S 另完成了严格 B1 全集复测：Mean KLD `0.313374`，same-top `82.3016%`。相对表中 B4 结果，Mean KLD 相差 `0.000186`，same-top 相差 `0.0630` 个百分点。B4 的四条序列相互独立，差异来自 M=2048 与 M=512 算子路径的浮点舍入。
