# MFQ 开发架构与算子参考

本文是 MFQ 开发的源码索引。修改文件格式、量化器、模型图、运行时分发、
CUDA kernel、KLD evaluator、张量并行或异构推理时，必须同步更新本文。
当前自动核对的 CUDA 入口共 234 条：MFQ Python 扩展 122 条、native-only
运行时 43 条、内置 TPQ 62 条、离线量化 7 条。

llama.cpp 的对照架构、GGML 全算子表、CPU/CUDA 分发、量化 recipe、imatrix
和 perplexity/KLD 协议见
[`llamacpp-developer-architecture-and-operator-reference.md`](llamacpp-developer-architecture-and-operator-reference.md)。

最后核对日期：2026-07-29

## 1. 维护规则

1. 区分三个层次：
   - 模型语义算子：Attention、GDN、FFN、MoE 等模型计算。
   - 运行时组合算子：`QuantLinearGroup`、`FFN::forward`、KV cache、路由和融合。
   - CUDA kernel：真正决定 dtype、累加方式、tile、workspace 和性能的实现。
2. 性能结论必须写明模型、GPU、`B/T/M/N/K`、权重格式、激活格式、
   kernel 路径、warmup、重复次数和同步边界。
3. 数值结论必须写明 reference、tokenization、chat template、context、
   batch/ubatch、chunk 数、scored token 数、KLD 方向和异常 chunk 处理。
4. `--kl-mmq` 是 KLD 对照专用覆盖项，不能代表默认生产分发。
5. `MFQ_*`/`TPQ_*` 环境变量改变实验条件。报告结果时必须保存环境。
6. 新增或删除 CUDA Python 绑定后运行：

   ```powershell
   python -m pytest -q -p no:cacheprovider tests/test_operator_reference.py
   ```

## 2. 仓库分层

| 层 | 主要位置 | 责任 |
|---|---|---|
| 文件与格式 | `mfq/formats/` | MFQ header、record、NINT/NVQ/NPQ/NEPQ/CCCP/NINTM pack/unpack |
| 量化算法 | `mfq/quantize/` | NINT、VQ/PQ、JSC、tensor codebook、imatrix、NEPQ、CCCP |
| 校准与分配 | `mfq/calibration/` | 数据集、统计量、候选误差、码率分配、REAP EW、端到端 refinement |
| Python kernel 封装 | `mfq/kernels/cuda/*.py` | CUDA 扩展加载、参数检查、workspace 与 Python 分发 |
| CUDA kernel | `mfq/kernels/cuda/*.cu` | packed GEMV/MMQ、Attention、GDN、MoE、KV、norm、sampling |
| Python runtime | `mfq/runtime/` | 参考/研究用模型图、量化 Linear/FFN、CCCP adapter |
| Native runtime | `cpp_runtime/mfq_decode.cpp` | 生产模型加载、模型图、分发、cache、KLD、profiling、CLI |
| HTTP 服务 | `cpp_runtime/mfq_server.cpp` | OpenAI 兼容 API、SSE、WebUI 静态资源 |
| 内置 TPQ | `mfq/_vendor/tpq/` | CCCP DeepSeek-V4 图、slot/cache、异构 residency、VQ kernel |
| 转换工具 | `mfq/tools/` | HF/GGUF/V4F/CCCP 到 MFQ、scheme 与 artifact |
| 测试 | `tests/` | 格式、量化、CUDA 数值、runtime、KLD、WebUI |

权威实现优先级：

| 问题 | 权威源码 |
|---|---|
| 文件位流 | `mfq/formats/*.py` |
| Python 量化结果 | `mfq/quantize/*.py` |
| Native 生产分发 | `cpp_runtime/mfq_decode.cpp` |
| CUDA 算术与 geometry | 对应 `.cu` 文件 |
| CCCP/TPQ 模型图 | `mfq/_vendor/tpq/` |
| llama.cpp 对照 | 对照 checkout 的 `src/models/` 与 `ggml/src/ggml-cuda/` |

## 3. 文件与权重表示

### 3.1 MFQ 容器

`MfqFile` 读取 `MFQ1`：

```text
magic "MFQ1"
version
scheme label
version >= 2: extra metadata key/value
record directory: name, dtype, nbytes
record payloads
```

Native loader 先读目录，按 tensor 读取 blob。基础 loader 仍用 `ifstream`；
Linux CPU/GPU 异构路径可在读取后对对应区间调用 `POSIX_FADV_DONTNEED`。

Overlay：

| 机制 | 环境变量 | 约束 |
|---|---|---|
| 完整 tensor 替换 | `MFQ_TENSOR_OVERLAY` | overlay record 必须替换已有名称 |
| routed expert delta | `MFQ_EXPERT_OVERLAY` | base 必须是 `NINTM`，overlay 必须是 `NINTMD` |

### 3.2 生产权重格式

| 家族 | 格式 | 关键几何 | Native Dense | NINTM | 主要 CUDA 文件 |
|---|---|---|---|---|---|
| NINT | NINT2 | 2 bit, gs16 | 是 | 是 | `nint_matmul.cu`, `moe.cu` |
| NINT | NINT3 | 3 bit, gs24 | 是 | 是 | 同上 |
| NINT | NINT4 | 4 bit, gs24 | 是 | 是 | 同上 |
| NINT | NINT5 | 5 bit, gs28 | 是 | 是 | 同上 |
| NINT | NINT6 | 6 bit, gs24；部分 lm-head 为 gs26 | 是 | 是 | 同上 |
| NINT | NINT8 | 8 bit, gs48；部分 gs24 | 是 | 是 | 同上 |
| NINT | NINT8-0 | 对称 INT8, gs32 | 是 | 是 | 同上 |
| NVQ | NVQ2/NVQ2J/NVQ2J-L/NVQ2J-XL | E8, 8D；8/10/12-bit index | 是 | 是 | `nvq_matmul.cu` |
| NVQ | NVQ3/NVQ3J/NVQ3J-512/NVQ3J-L | D4, 4D；8/9/10-bit index | 是 | 是 | 同上 |
| NVQ | NVQ1-L/NVQ1-S | ternary VQ | 是 | 是 | 同上 |
| NPQ | NPQ0-L/NPQ0-S | 约 1 bit/weight | 是 | 是 | 同上 |
| NEPQ | NEPQ1-L/S, NEPQ0-L/S | 跨专家 bank + signed Hadamard | Dense kernel 存在，主要用于 MoE | 是 | `nepq.cu`, `nvq_matmul.cu` |
| CCCP | x/v/w/vv | D=8/4/2，固定 index dtype | TPQ dense 为 INT4-G64 | CCCP pool | `_vendor/tpq/csrc/vq_gemv.cu` |

`NINTM v2` 使用 `NIM2`。同一 `family + profile + artifact + rotation`
的 experts 组成 cohort。运行时保持 packed payload，不为每一行保存格式标签。

### 3.3 精度与格式的边界

| 项目 | 是否是新格式 | 改变内容 |
|---|---:|---|
| imatrix | 否 | 量化误差权重 |
| neuron gain | 否 | 逐输出行目标缩放 |
| tensor-wise codebook | 否 | VQ/PQ 码本 |
| REAP/EW | 否 | expert precision 分配 |
| soft refinement | 否 | 端到端率失真分配 |
| NINTM | 容器 | 一个逻辑 expert tensor 内混合格式 |
| NINT8-1 | 激活临时格式 | llama.cpp MMQ 对照用 Q8_1 activation block |

### 3.4 量化工件与数据契约

| 工件 | 定义位置 | 关键内容 | 消费者 |
|---|---|---|---|
| calibration corpus v1 | `calibration/dataset.py` | 固定 token IDs、split、domain、source hash | statistics/refinement |
| calibration statistics v1 | `calibration/statistics.py` | `E[x_i²]`、逐输出行 Fisher、采样数、模型身份 | candidate evaluator |
| calibration scheme | `calibration/artifact.py` | tensor/group/expert precision、预算、来源元数据 | HF/GGUF/V4F converter |
| imatrix | `quantize/imatrix.py` | 输入通道二阶矩；支持 GGUF 和 legacy | NINT/NVQ/NPQ quantizer |
| row importance | `quantize/row_importance.py` | 每个输出行的非负权重 | gain/sensitivity 工具 |
| tensor codebook | `quantize/*tensor_codebook.py` | 码本、训练配置、数据与 tensor identity | NVQ/NPQ converter |
| REAP expert table | `calibration/reap_expertwise.py` | layer/expert frequency、probability、REAP、exposure | expert precision allocator |
| NINTM runtime metadata | `formats/io.py` | 当前用于 NEPQ rotation sign vector | native/Python loader |
| MFQ shard metadata | `formats/shards.py` | `split.no/count/tensors.count/records.count`、编号与规划 | HF/GGUF converter、Python/C++ loader |

imatrix entry 可为一条 `[K]` 输入通道向量，也可为 expert-wise `[E,K]`。
三维专家张量按扁平行号 `row // rows_per_expert` 选择对应 expert 的矩阵。
NINT 的 imatrix 路径使用 llama.cpp 风格 256-weight `sigma²` block：

```text
element_weight = importance * sqrt(2*mean(W_block²) + W²)
```

随后组内使用 weighted affine search，顶层 scale/min 使用 weighted
`make_qp`；weight-only 路径继续使用原 `make_qkx2` 与最大值锚定。两条路径
不能仅凭同一个 `NintSpec` 视为数值等价。

### 3.5 离线量化执行链

```text
source index / GGUF directory
 -> tensor name + logical shape plan
 -> UD recipe / explicit overrides / expert scheme
 -> imatrix and codebook artifact binding
 -> row-stream quantization
 -> per-record temporary blob and exact byte-count check
 -> 单文件或编号分片的 MFQ directory + payload assembly
 -> reopen/shape/dtype/dequant verification
```

| 输入 | 主入口 | 特有步骤 |
|---|---|---|
| HF safetensors | `mfq.tools.quantize_hf_to_mfq` | HF→GGUF 名称映射、Qwen linear-attention split、GLM derived tensors |
| GGUF | `mfq.tools.quantize_gguf_to_mfq` | 读取原 recipe、可直接映射 Q/IQ→NINT/NVQ、绑定 GGUF imatrix |
| DeepSeek V4F | `mfq.tools.quantize_v4f_to_mfq` | shard 到达即训练/量化、tiered expert artifacts |
| TPQ directory/V4F | `import_tpq_to_mfq` / `quantize_tpq_v4f_to_mfq` | 保持原 PQ index/codebook 或直接训练 TPQ tier |
| overlay | `materialize_mfq_overlay` / `upgrade_v4f_mfq` | 只重写选中的 record/expert stream |

转换器必须先计算每个 blob 的目标字节数。临时 blob 长度、写入长度和最终
record directory 中的 `nbytes` 任一不一致都应停止，不能依赖最终加载时才发现。
`mfq quantize INPUT OUTPUT` 是稳定的统一入口。它自动识别 HF safetensors
目录与 BF16 GGUF 文件，将 `--recipe`、`--scheme`/`--ew-scheme`、
`--imatrix`、IN、分片及恢复参数规范化后，直接调用上表中的生产转换器；
底层模块入口继续用于量化算法实验。

## 4. Native 模型图

### 4.1 顶层

```text
ids
 -> quantized embedding lookup
 -> optional Gemma embed scale
 -> blocks[0..L-1]
 -> optional DSV4 HC head merge
 -> final RMSNorm
 -> quantized LM head
 -> logits / sampler
```

`Model::hidden_forward` 的输入为 `[B,T]`。Linear 的逻辑矩阵行数
`M = B*T`。KV/GDN state 只在 batch shape 改变或显式 reset 时重新分配。

### 4.2 架构表

| 架构 | `model_type`/layer type | Block | Attention | FFN/MoE |
|---|---|---|---|---|
| Qwen3.5/3.6 | `full_attention` | `FullBlock` | GQA + q/k norm + RoPE + KV | Dense SwiGLU 或 routed + shared expert |
| Qwen3.5/3.6 | `linear_attention` | `LinearBlock` | q/k/v projection + causal conv + GDN | Dense SwiGLU |
| Gemma4 | `gemma4(_text)`，full/sliding | `FullBlock` | global 或 SWA，额外 post norms/layer scale | Dense GeGLU 与 routed GeGLU 双路径 |
| DeepSeek-V4 | `deepseek_v4` | `Dsv4Block` | HCA/CSA/mHC、compressed pool、indexer | sqrt-softplus/hash routed MoE + shared expert |
| GLM DSA | `glm_moe_dsa` | `GlmDsaBlock` | MLA/DSA，full/shared indexer | dense 或 sparse MoE schedule |

### 4.3 Qwen Full Attention

```text
RMSNorm
 -> grouped Q/K/V projections
 -> Q gate split (Qwen3.5/3.6)
 -> Q/K(/V) norm
 -> RoPE
 -> KV write
 -> prefill/decode attention
 -> optional sigmoid Q gate
 -> output projection
 -> residual + FFN RMSNorm
 -> FFN
 -> residual
```

Prefill Attention 分发：

| 条件 | 路径 |
|---|---|
| D=512, Q:KV=8:1, non-sliding, T%256=0 | llama-style Flash512 |
| D=256, Q:KV=2:1, sliding, T>=32 | llama-style Flash256 SWA |
| D=256, Q:KV=4:1, non-sliding, T>=32 | llama-style Flash256 |
| sliding 但不满足专用条件 | MFQ SWA kernel |
| 其他 GQA | ATen SDPA，`enable_gqa=true` |

Qwen3.6-27B 的 full Attention 为 24Q/4KV、D=256，比例 6:1，
当前不进入 MFQ 的 4:1 专用 Flash256，落到 ATen SDPA。

### 4.4 Qwen Linear Attention/GDN

```text
RMSNorm
 -> QK/V/Z/A/B projections
 -> fused alpha/beta transform
 -> fused prefill conv + q/k L2（T>=256）
 -> GDN
 -> per-head RMSNorm + Z gate
 -> output projection
 -> residual + FFN RMSNorm
 -> FFN
 -> residual
```

Qwen3.6 GGUF 的 alpha/beta 为 dense tail；qkv/z 仍可量化。
默认 GDN 使用 transposed state 和 warp-column kernel。D=128 时：

| 实现 | grid | block | state |
|---|---|---|---|
| MFQ tiled GDN | `(B*Hv, ceil(D/4), 1)` | `(32,4,1)` | 每 warp 一列，寄存器跨 T |
| llama.cpp GDN | `(Hv, n_seqs, ceil(D/4))` | `(32,4,1)` | 每 warp 一列，寄存器跨 T |

两者 geometry 和状态生命周期基本对位；需要 profile 后才能判断剩余地址计算差异。

### 4.5 FFN

| 场景 | gate/up | activation | down |
|---|---|---|---|
| Decode M=1、同 NINT layout | grouped packed GEMV | fused SwiGLU/GeGLU | packed GEMV |
| Decode M=1、兼容 NVQ | two-projection GEMV | fused | NVQ GEMV |
| 可直接量化中间激活 | gate/up + activation + quantize | 不生成 FP16 中间 | down 从 qx |
| Prefill M>1 | grouped 或独立 matmul | FP16/FP32 pointwise | matmul |
| Tensor parallel | 每卡 gate/up + activation + local down | 本卡 | hidden-size partial reduce |

`--kl-mmq fp16/nint8_1` 会绕过默认 NINT/NVQ 分发。当前实现下，
prefill gate/up 可能按两个 projection 分别调用 common MMQ。

### 4.6 Routed MoE

```text
FP16 hidden -> FP32 router
 -> top-k / hash route
 -> compact route map
 -> optional expert cache prefetch
 -> grouped gate_up
 -> SwiGLU/GeGLU
 -> grouped down
 -> route-weight reduce
 -> shared expert gate/combine
```

| 能力 | 实现 |
|---|---|
| mixed precision experts | NINTM cohort |
| activation reuse | 相同 transform/gs/group count 的 cohort 复用 |
| NEPQ rotation reuse | 相同 rotation key 复用 Hadamard activation |
| CPU/GPU 异构 | 压缩 expert CPU resident + GPU LRU arena |
| profile 预热 | ranking 按层分配；frequency 可做全局预算 |
| history prefetch | 上一 token 的逐层 route，默认关闭 |
| tensor parallel | expert output-axis 切分；gate/up 配对；down partial reduce |

## 5. Dense 量化矩阵分发

### 5.1 NINT 默认生产分发

| 格式/条件 | M | 路径 |
|---|---:|---|
| NINT8-0 | `<=8` | q8 activation GEMV |
| NINT8-0 | `9..64` | packed MMQ |
| NINT8-0 | `>=65` | full dequant + GEMM |
| NINT2 gs16，N>=2048 | `9..64/128` | group32 int8 MMA；上限随 N |
| NINT3 gs24 | `>=9`，默认 | group32 int8 MMA |
| NINT3 gs24 | `>=9`，`MFQ_NINT3_PREFILL_PATH=f16` | online FP16 MMA |
| NINT4 gs24 | `16/32` | group32 int8 MMA |
| NINT4 | `1` | packed GEMV |
| NINT4 | 小 M | batched packed GEMV |
| NINT4 | 大 M | compact dequant + cuBLAS |
| NINT5/6/8 | `<=8` | generic packed-bits GEMV |
| NINT6 gs24 | `16..32` 且 N<=8192 | FP16 MMA split-K=4 |
| NINT8 | `<=64` 或强制 | packed u8 MMQ |
| NINT5/6/8 | 大 M | compact dequant + GEMM |

阈值在 `nint_matmul()`。相关变量包括
`MFQ_NINT4_BATCH_GEMV_MAX_M`、`MFQ_NINT_HI_GEMV_MAX_M`、
`MFQ_NINT6_BATCH_GEMV`、`MFQ_NINT8_PREFILL_MMQ`。

### 5.2 NVQ/NPQ 默认生产分发

`select_nvq_matmul_path()` 返回：

| 路径 | 含义 |
|---|---|
| `Gemv` | q8 activation，在线 codebook lookup，M 很小 |
| `Mmq` | gs24 int8 Tensor Core |
| `OnlineF16` | 在线解包进 shared tile，FP16 Tensor Core |
| `DequantGemm` | 完整临时反量化后 GEMM |

具体阈值依赖 format、M、N/K expansion ratio。开发时不能只按“NVQ bit 数”
推断实际 kernel。

### 5.3 KLD common MMQ 覆盖

| `--kl-mmq` | 激活 | NINT/NVQ 权重计算 | 用途 |
|---|---|---|---|
| `default` | 生产路径各自决定 | 默认分发表 | 实际 runtime |
| `fp16` | FP16 | 在线解包 FP16 MMA | 取消激活量化的权重格式对照 |
| `nint8_1` | 先量化为 Q8_1，再重建 FP16 | 与 `fp16` 相同的在线解包 FP16 MMA | 激活量化误差诊断；当前不能作为 llama.cpp 整数 MMQ 的公平对照 |

Common FP16 NINT kernel：

```text
grid.x = ceil(N/64)
grid.y = ceil(M/(16*MTILES))
grid.z = split_k
block  = (32,8)
MTILES = 1/2/4/8（NINT5 大 M 固定 4）
split_k = 1..4（按 SM 数、输出 tile 数和 K chunk 数选择）
```

每个 K chunk 只在 shared memory 中保存当前 `W_s` 和 `X_s`，计算后覆盖。
它不会把完整模型或完整矩阵永久展开为 FP16。

llama.cpp 在 Ada 及更新 NVIDIA GPU 上，`ggml_cuda_should_use_mmq()` 对支持的
Q/IQ 权重返回 true；激活由 `quantize_mmq_q8_1_cuda()` 转成 Q8_1 后进入整数 MMA。
MFQ 当前的 `nint8_1` 模式只复现 Q8_1 激活量化与重建，尚未接入相同的整数 MMA。

### 5.4 Dense CUDA 算子几何

约定 `M` 为 activation rows，`N` 为输出行，`K` 为输入宽度，`ng=ceil(K/gs)`。

| 家族/算子 | 输入与输出 | activation 处理 | 主 kernel geometry | 适用范围 |
|---|---|---|---|---|
| NINT packed GEMV | `[M,K] -> [M,N]` | 每 `(row,group)` 对称 INT8，并保存 scale/sum | 常规 `grid=(ceil(N/8),M)`, `block=(32,8)`；专用 gs24 为 `grid=(N,M)`, `block=(32,warps)` | decode/small M |
| NINT generic bits GEMV | 同上 | 同上 | 每 block 处理少量输出行；warps 依 bits/gs/profile | NINT2/3/5/6/8 |
| NINT group32 MMA | 同上 | gs24 直接排为 K32；NINT2 将两个 gs16 拼成 K32 | `grid=(ceil(N/64),ceil(M/(16*MTILES)),split_k)`, `block=(32,8)` | NINT2/3/4 中 M |
| NINT online FP16 MMA | 同上 | FP16 activation；当前 K chunk 解包进 shared tile | `grid=(ceil(N/64),ceil(M/(16*MTILES)),split_k)`, `block=(32,8)`；1–4 路 FP32 partial 归并 | NINT2/3/4/5/6/8 及 KLD |
| NINT Q8_1 重建诊断 | 同上 | Q8_1 量化后重建 FP16 | 使用 NINT online FP16 MMA | KLD 激活量化误差诊断 |
| NINT full dequant + GEMM | `[packed N,K] -> f16 [N,K]` 后 GEMM | 无 activation quantization | dequant 256-thread grid；GEMM 用 ATen/cuBLAS | 在线 MMQ 未覆盖的 profile 或中间 M |
| NVQ/NPQ GEMV | `[M,K] -> [M,N]` | 每 gs24 INT8 | `grid=N`, `block=(32,2/4/8)`，warp 数按 format/N/M | `M<=16` 候选 |
| NVQ/NPQ int8 MMA | 同上 | 每 `(M,ng)` 一 warp 量化 | `grid=(ceil(N/64),1)`, `block=(32,8)` | `M=4..64` |
| NVQ/NPQ online FP16 | 同上 | FP16；一个 `64 x (4*gs)` weight tile 在线解码 | `grid=(ceil(N/64),ceil(M/(16*MTILES)),split_k)`, `block=(32,8)`；1–4 路 FP32 partial 归并 | 中/大 M |
| NVQ/NPQ Q8_1 重建诊断 | 同上 | Q8_1 量化后重建 FP16 | 使用 NVQ/NPQ online FP16 | KLD 激活量化误差诊断 |
| NEPQ GEMV | `[M,K] -> [M,N]` | gs24 INT8 + rotation 后输入 | `grid=N`, `block=(32,2/4/8)` | `M=1..16` |
| NEPQ int8 MMA | 同上 | gs24 INT8 | `grid=(ceil(N/64),1)`, `block=(32,8)` | `M=4..64` |
| NEPQ online FP16 | 同上 | 在线 bank/table lookup | `grid=(ceil(N/64),ceil(M/(16*MTILES)))`, `block=(32,8)` | `M=16..256` |
| embedding decode | token IDs -> `[tokens,K]` | 只解码被索引的行 | 256 threads，grid 按 token×packed-vector | NINT/NVQ/NINT8-0 |

`MTILES` 表示一个 block 处理多少个 16-row MMA tile。NINT common FP16 和
NVQ online FP16 都只保留当前 K chunk 的 shared state。full dequant 路径才生成
完整 `[N,K]` 临时权重。

### 5.5 NVQ 运行时精确选择条件

`kernel_format` 的内部编号与公开名称：

| 编号 | 格式 |
|---:|---|
| 1/8 | NVQ1-L / NVQ1-S |
| 2/4 | NVQ2 / NVQ2 execution layout |
| 5/6 | NVQ2J / NVQ2J execution layout |
| 3 | NVQ3 |
| 10/11/12 | NVQ3J / 2-bank variant / NVQ3J-512 |
| 13/14 | NVQ2J-L / NVQ2J-XL |
| 15 | NVQ3J-L |
| 7/9 | NPQ0-L / NPQ0-S |

默认规则的关键边界：

| 条件 | 选择 |
|---|---|
| 一般格式 `M<=13` | GEMV |
| 一般格式 `M=14` | `N>=2048` 用 GEMV，否则 dequant+GEMM |
| E8 family `M=15, N>=8192` | int8 MMA |
| E8 family `M=16, N>=6144` | int8 MMA |
| E8 wide expansion `N>=3K`，`M=17..31/33..47` | online FP16 |
| E8 `M=32,N>=8192` 或 `M=48,N>=12288` | int8 MMA |
| NVQ3J-512 | 另有按 `N/K` 比例细分的 GEMV/MMQ/online FP16 规则 |
| 其余 | full dequant + GEMM |

这些阈值来自 `select_nvq_matmul_path()`，属于按实测固化的 shape policy。
修改 kernel 后必须重新扫交叉点，不能只改 CUDA 实现而保留旧 policy。

## 6. Attention、KV、GDN 与状态

| 模块 | Prefill | Decode | 持久状态 |
|---|---|---|---|
| Full Attention | llama-style Flash/MFQ SWA/ATen SDPA | llama-style split-K 或 ATen | FP16 KV |
| Sliding Attention | causal window kernel | circular KV kernel | ring KV |
| Qwen GDN | fused conv prefill + warp-column scan | fused conv decode + scan | FP32 conv/GDN state |
| DSV4 compressed pool | pool update + attention plan | incremental pool update | pool/state/gate |
| GLM DSA | indexer top-k + sparse attention | cache/index state | latent KV/index K |

KV 写入有线性和 ring 两组 kernel。`MFQ_KV_CACHE_WRITE_ATEN=1` 可切换诊断路径。

### 6.1 Attention 算子表

| 算子 | 固定/允许形状 | grid / block | 切分与临时状态 | 生产调用 |
|---|---|---|---|---|
| generic full/GQA | `q=[B,Hq,T,D]`, `D<=512` | 每 `(B,Hq,query)` 一个 block；block 为 64/128/256/512 threads | 总 query `<128` 且 visible keys `>=512` 时，K 每 256 切分，最多 16 parts | 非专用 full attention |
| generic SWA | 同上 + window | 同 generic | circular/linear visible range；可 planned length | sliding fallback |
| llama Flash256 | `D=256`, `Hq/Hkv=4` | llama helper 决定 warps/shared；tile 由 `ncols1*ncols2` 表示 | `T%64=0` 用 `<256,256,16,4>`，其他长度用 8/16/32/64-column 小模板 | Q:KV=4:1 full |
| llama Flash256 SWA | `D=256`, `Hq/Hkv=2` | 同上 | 对齐时 `<256,256,32,2>` | Q:KV=2:1 sliding |
| llama Flash512 | `D=512`, `Hq/Hkv=8`, `T%64=0` | `<512,512,8,8>` | llama Flash MMA | Q:KV=8:1 full |
| llama decode | D256/D512 | `block=(32,nwarps)`；block 数由 occupancy 和 KV tiles 决定 | Ada 或 tile efficiency `<75%` 默认 stream-K；多 block/tile 用 O/M/L fixup | 专用 decode |
| DSV4 sparse | `H=64,D=512`, selected 为 32 的倍数 | `block=(32,nwarps)`；occupancy 驱动 block 数 | `ncols1=1,ncols2=16`；多 block/tile 用 uniform fixup | HCA/CSA sparse pool |
| GLM sparse MLA | `H=64,Dq=576,Dv=512` | 同 DSV4 | `ncols1=1,ncols2=16`，stream-K fixup | DSA top-k path |
| GLM dense/cached MLA | `Dq=576,Dv=512` | llama Flash MMA helper | cached prefill 用 1/2/4 query-column variant | full/shared layers |

通用 attention 的在线 softmax在 FP32 中维护 max/sum。llama-style、DSV4 和
GLM sparse 路径的 dynamic shared memory、warps、KV batch size均由同一套
`ggml_cuda_fattn_mma_get_*` helper 决定。修改 head dimension 或 GQA 比例时，
必须同步验证 template 条件和 runtime dispatch。

### 6.2 DSV4 压缩与 GLM indexer

| 算子 | 输入/输出 | geometry | 关键约束 |
|---|---|---|---|
| DSV4 FP4 simulation | f16，末维 32 对齐 | 每 32 值一个 32-thread block | 数值模拟，不改变存储 dtype |
| DSV4 compressor | `[B,W,ratio,D/2D] -> [B,W,D]` | `grid=B*W`, `block=D` | `D=128/512`, ratio `<=128`，可 overlap |
| DSV4 decode pool update | token + remainder state -> pool | `grid=B`, `block=D` | 原位更新 remainder/previous/pool/seq_len |
| DSV4 indexer score | `64 x 128` heads 对 pool K | `grid=(ceil(K/64),M,B)`, `block=(32,16)` | 输出 `[B,M,K]` f16 |
| DSV4 top-512 | `[B,M,K] -> ids` | 每 row 一个 256-thread block | 固定输出 512 个 int32 index |
| DSV4 prefill/decode plan | top-k -> index+mask | 最多 65535 个 256-thread blocks | local window 与 compressed pool 合并，selected 向 32 对齐 |
| GLM interleaved RoPE | f16/f32 `[B,H,T,D]` | 最多 65535 个 256-thread blocks | rotary dim 为偶数 |
| GLM indexer LayerNorm | f16 `[...,128]` | 每 row 一个 128-thread block | weight/bias FP32 |
| GLM indexer score | 32 heads × 128 | `grid=(ceil(K/64),M,B)`, `block=(32,8)` | FP32 score；decode 接收逐 batch seq_len |
| GLM cache write | f16 `[B,T,D]` | 最多 65535 个 256-thread blocks | 按 int64 position 写入 |

### 6.3 GDN、HC、KV 与基础算子

| 算子家族 | 输入/状态 | geometry | 数值/生命周期 |
|---|---|---|---|
| GDN warp-column | q/k/v/g/beta FP32；state `[B,Hv,D,D]` FP32 | `grid=(B*Hv,ceil(D/4),1)`, `block=(32,4)` | 默认；支持 `D=32/64/128`、KDA、tiled Q/K heads、transposed state |
| GDN column | 同上 | `grid=B*H*D`, `block=128` | 仅 `T<=4` 且显式启用；要求 Hq=Hv |
| GDN shared-state | 同上 | `grid=B*H`, `block=D`, dynamic shared=`D²*4` | 要求 Hq=Hv；超过 48 KiB opt-in |
| DSV4 HC pre | f16 `[B,T,4,4096]` + f32 mixes `[B,T,24]` | 每 token row 一个 256-thread block | warp 内 20 次 4x4 Sinkhorn，输出 reduced/post/combination |
| DSV4 HC post | f16 residual `[B,T,4,4096]` | `grid=(ceil(4096/256),B*T,4)`, `block=256` | FP32 residual mix，输出 f16 |
| RMS/L2 norm | 每行 | 每 row 一个 256-thread block | FP32 reduction；另有 f16 I/O 版本 |
| residual/activation | elementwise 或每 row | 256 threads，grid 通常上限 4096 | 包含 residual+norm、SiLU/GELU、Gemma merge |
| RoPE | `[B,H,T,D]` | `grid=B*H*T`, `block=256` | rotate-half/partial/MRoPE/table variants |
| KV write | linear/ring cache | 256 threads，grid 上限 4096 | ring variant支持显式 positions |
| Qwen conv/GDN prep | q/k/v conv state | prefill 按 `B*T`/head tile；decode 256 threads | prefill 可融合 conv+SiLU+Q/K L2 |
| sampling | logits/counts | 256 threads | top-k 最多 1024；radix fast path top-k `<=64` |

### 6.4 MoE 算子与调度表

约定 `tokens=T`，每 token 选择 `routes=R`，expert 输出宽度为 `O`。

| 阶段/条件 | 算子与 geometry | 说明 |
|---|---|---|
| top-k 通用 | `grid=ceil(T/4)`, `block=(32,4)` | top-k `<=16`；支持 softmax/sigmoid/sqrt-softplus |
| top-k decode 专用 | `1x128,k=8`: 1 warp；`1x256,k=8/6`: 256 threads | Qwen/Gemma/DSV4 路由热路径 |
| compact route map | count + 单线程 prefix scan + tile fill + scatter | `T>8` 的 grouped kernel 前置步骤 |
| activation quantize | `grid=(input_rows,groups)`, block 向上取整到 warp | gs16/24/28/48；gs24+28 可单 kernel 双量化 |
| NINTM hetero decode `T=1` | `grid=(O,R)`, `block=(32,4)`；K<1024 时按 4 行/warp | 一个 dispatch 跨 NINT cohorts |
| NINTM hetero `T=2..8` | `grid=(ceil(O/2),R)`, `block=(32,2/4/8)` | token tile 与 T 对应 |
| homogeneous NINT `T=9..64` | direct token-warp MMVQ，默认 16 warps | 可设 8/16/32 warps |
| homogeneous NINT gs24/28 `T=16..128` | group32 MMA，`block=(32,8)` | `MFQ_MOE_SMALL_MMQ` 控制 |
| NINTM FP16 prefill | persistent MMA，`block=(32,8)`, BN=64, BM=16/32/64 | 默认 `T>=256`；block 数上限默认 4096 |
| homogeneous NINT grouped | route tile=8，`block=(32,4)` | `T<=32` 默认 persistent，最多 4096 blocks |
| NINT8-0 prefill | FP16 MMA，BN=64，BM=16/32/64 | 可绕过 activation INT8 |
| NVQ/NEPQ `T<=8` | `grid=(ceil(O/rows_per_block),T*R)`, `block=(32,warps)` | NVQ 可启用 ordered exact reduction |
| NVQ/NEPQ `T>8` | route tile=8，`block=(32,4)`，最多 4096 blocks | compact route map |
| route reduce | `[T,R,O] + weights -> [T,O]` | 256-thread pointwise/reduction |
| shared gate reduce | routed reduce + shared expert gate | 融合版本避免额外输出遍历 |

NINTM prefill 的默认优先级：

```text
KLD override
 -> T >= 256 且 route map 可用：heterogeneous online-FP16 MMA
 -> T <= 4（T=1 永远允许）：heterogeneous q8 activation path
 -> 按 cohort 的 homogeneous pool path
```

同一个 activation 只可在 `transform + gs + group_count + device` 完全一致时复用。
gate/up 和 down 的输入不同，不能跨投影复用。NEPQ rotation key 不同时也不能复用。

## 7. 并行、内存与生命周期

### 7.1 Tensor parallel

| 权重角色 | 切分轴 | 汇合 |
|---|---|---|
| Q/K/V、gate/up、LM head | output rows | concat |
| attention output、FFN down | input groups | FP32 partial sum 后转回输出 dtype |
| embedding | mirrored | 无 |
| routed gate/up | paired output rows | gate 半区、up 半区分别 concat |
| routed down | output rows/本地 partial | primary device 汇合 |

切片必须满足 packed group 与 kernel alignment。NINT/NVQ input-axis 以 group 为单位切。

### 7.2 Workspace

| 家族 | 主要 workspace |
|---|---|
| NINT GEMV | `qx`, `xscale`, `xsum` |
| NINT MMQ | `mmq_qx`, optional split-K partial |
| NVQ | `qx`, `xscale`, optional SwiGLU scratch |
| MoE | route map、pool activation workspace、heterogeneous pointer arrays |
| Attention decode | partial O/M/L、mask、kv_max、metadata |
| Sampling | random、counts、argmax partial |

Workspace 以 shape/M 缓存。`torch::empty` 是 CUDA caching allocator 分配，
不能仅凭调用次数判断每层发生系统级分配；需要 profiler 和 allocator 统计共同验证。

### 7.3 CPU/GPU expert cache

缓存单位是完整 expert bundle。LRU key 包含 source/cohort/expert。一次 miss 会：

1. 获取或替换 arena slot；
2. 复制该 expert 的所有字段；
3. 更新 global-expert 到 local-slot map；
4. 在消费 stream 前等待 transfer event。

预取和 demand 的命中/miss 分开计数。只允许管理 run contract 记录的进程和 cache 工件。

### 7.4 TPQ/CCCP expert residency

`CCCPStore` 读取 `cccp.json`、dense shard 和逐层 expert shard。expert 的 GU/DN
索引保持 `uint8/uint16` packed 状态；同层同档 codebook 共享。`ExpertPool` 的层级为：

```text
disk shard
 -> optional RAM LRU / permanently resident RAM
 -> optional page-locked host copy
 -> pinned staging slots + asynchronous DMA
 -> fixed GPU arenas by ExpertSignature
 -> slot VQ kernel
```

| 组件 | key/单位 | 驱逐与同步 | 开发约束 |
|---|---|---|---|
| RAM LRU | `(layer, expert)` 完整 GU+DN | OrderedDict LRU | miss 才读 shard；并行读取默认 12 workers |
| pinned experts | profile 每层热度排名 | 永不驱逐 | `TPQ_PIN_GB` 只控制热专家；全量 RAM 常驻另由 `TPQ_FULL_RESIDENT` 控制 |
| pinned staging | 默认 32 个、每槽 12 MiB | CUDA event 完成前不可复用 | direct pinned source 可绕过 CPU memcpy |
| GPU arena | `ExpertSignature`：GU/DN shape、dtype | slot lease/LRU；in-flight slot 不可驱逐 | 各 signature 按模型占比和 top-k 下限分配 |
| GPU full resident | 所有非 drop expert | 永不替换 | arena 容量覆盖每种 signature 才能启用 |
| codebook cache | `(layer, tier)` | 常驻 device cache | codebook 不随每次 expert miss 重传 |

一次异步 miss 的顺序为：占用 slot、标记 in-flight、复制索引、记录 event、消费
stream 等待 event、清除 in-flight。任何把 slot 提前交还的改动都可能让 DMA 覆写
正在使用的 expert。`get_many` 必须在同一把 staging lock 下批量命中、加载和租约，
避免每个 expert 重复获取锁。

预取使用上一 token 的每层路由作为预测。全量 GPU 常驻时自动关闭；全量 RAM 常驻
时默认也关闭，避免 staging 与 demand 竞争。动态显存监视器按物理空闲显存做滞回：
默认低于 0.8 GiB 时将 cache budget 减少 0.5 GiB，高于 3.0 GiB 时逐步恢复，
查询周期 3 秒。

## 8. 评测路径

| Evaluator | 输入 | 主要特点 |
|---|---|---|
| historical/legacy | 历史逐步逻辑 | 用于复现旧记录 |
| optimized | 完整 context chunk | GPU 累积 KLD/CE/same-top，减少回读 |
| llama.cpp reference | GGUF + llama-perplexity/导出 logits | 必须记录 n_ctx/n_batch/n_ubatch/n_seq |

当前 Qwen3.5/3.6 历史全集协议：

```text
n_ctx=2048
n_batch=2048
n_ubatch=512（llama.cpp）
n_seq=1
145 chunks
148335 scored tokens
```

MFQ optimized evaluator 当前把 `[1,2048]` 一次送入模型，因此 Linear 看见
`M=2048`。llama.cpp 将同一 context 切成四个 ubatch，Linear 看见 `M=512`。
对比吞吐时必须明确这项 geometry 差异。

## 9. Native CUDA 绑定全表

下面各份清单由测试与源码 `m.def()` 强制保持一致。说明字段以 kernel 家族表为主，
精确 signature 以 `mfq/kernels/cuda/mfq_cuda.cpp` 声明为准。

### 9.1 MFQ CUDA 扩展（122）

<!-- MFQ_NATIVE_BINDINGS_BEGIN -->
- `acc_cuda` - MFQ CUDA Python binding.
- `acc_rms_norm_cuda` - MFQ CUDA Python binding.
- `acc_rms_norm_f16_cuda` - MFQ CUDA Python binding.
- `attention_cache_decode_cuda` - MFQ CUDA Python binding.
- `attention_cache_swa_cuda` - MFQ CUDA Python binding.
- `attention_cuda` - MFQ CUDA Python binding.
- `attention_swa_cuda` - MFQ CUDA Python binding.
- `decode_graph_commit_cuda` - MFQ CUDA Python binding.
- `embedding_lookup_cuda` - MFQ CUDA Python binding.
- `gdn_cuda` - MFQ CUDA Python binding.
- `gdn_inplace_cuda` - MFQ CUDA Python binding.
- `gdn_inplace_transposed_cuda` - MFQ CUDA Python binding.
- `gelu_mul_cuda` - MFQ CUDA Python binding.
- `gemma4_attn_residual_pre_norms_f16_cuda` - MFQ CUDA Python binding.
- `gemma4_ffn_merge_f16_cuda` - MFQ CUDA Python binding.
- `kv_cache_write_cuda` - MFQ CUDA Python binding.
- `kv_cache_write_ring_cuda` - MFQ CUDA Python binding.
- `kv_cache_write_ring_positions_cuda` - MFQ CUDA Python binding.
- `l2_norm_cuda` - MFQ CUDA Python binding.
- `linear_conv_qkv_decode_cuda` - MFQ CUDA Python binding.
- `moe_add_shared_gate_cuda` - MFQ CUDA Python binding.
- `moe_apply_expert_scale_cuda` - MFQ CUDA Python binding.
- `moe_build_expert_map_cuda` - MFQ CUDA Python binding.
- `moe_geglu_split_cuda` - MFQ CUDA Python binding.
- `moe_sqrtsoftplus_weights_cuda` - MFQ CUDA Python binding.
- `moe_swiglu_split_cuda` - MFQ CUDA Python binding.
- `moe_topk_cuda` - MFQ CUDA Python binding.
- `moe_weighted_reduce_cuda` - MFQ CUDA Python binding.
- `moe_weighted_reduce_shared_gate_cuda` - MFQ CUDA Python binding.
- `nepq_dequant_cuda` - MFQ CUDA Python binding.
- `nepq_gemm_f16_cuda` - MFQ CUDA Python binding.
- `nepq_gemv_ws_cuda` - MFQ CUDA Python binding.
- `nepq_hadamard_input_cuda` - MFQ CUDA Python binding.
- `nepq_mmq_ws_cuda` - MFQ CUDA Python binding.
- `nepq_moe_grouped_matmul_pool_ws_cuda` - MFQ CUDA Python binding.
- `nepq_moe_grouped_matmul_ws_cuda` - MFQ CUDA Python binding.
- `nint5_gs28_q5_argmax_ws_cuda` - MFQ CUDA Python binding.
- `nint5_gs28_q5_dequant_cuda` - MFQ CUDA Python binding.
- `nint5_gs28_q5_gemv_ws_cuda` - MFQ CUDA Python binding.
- `nint5_gs28_q5_repack_cuda` - MFQ CUDA Python binding.
- `nint8_one_quantize_reconstruct_cuda` - MFQ CUDA Python binding.
- `nint8_zero_mmq_f16_packed_cuda` - MFQ CUDA Python binding.
- `nint_cublas_gemm_nt_f16acc_cuda` - MFQ CUDA Python binding.
- `nint_dequant_full_packed_compact_bits_cuda` - MFQ CUDA Python binding.
- `nint_dequant_full_packed_compact_cuda` - MFQ CUDA Python binding.
- `nint_dequant_full_packed_cuda` - MFQ CUDA Python binding.
- `nint_dequant_full_packed_gs24_x2_cuda` - MFQ CUDA Python binding.
- `nint_dequant_full_packed_gs24_x2h2_cuda` - MFQ CUDA Python binding.
- `nint_dequant_full_packed_h2_cuda` - MFQ CUDA Python binding.
- `nint_dequant_wq_packed_cuda` - MFQ CUDA Python binding.
- `nint_embedding_lookup_cuda` - MFQ CUDA Python binding.
- `nint_embedding_lookup_packed_compact_bits_cuda` - MFQ CUDA Python binding.
- `nint_embedding_lookup_packed_compact_cuda` - MFQ CUDA Python binding.
- `nint_embedding_lookup_packed_eff_cuda` - MFQ CUDA Python binding.
- `nint_ffn_gate_up_geglu_quant_ws_cuda` - MFQ CUDA Python binding.
- `nint_ffn_gate_up_swiglu_quant_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_batch_eff2_gate_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_batch_eff2_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_batch_eff_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_batch_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_bits_argmax_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_bits_gate_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_bits_geglu_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_bits_linear_out_norm_gate_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_bits_m1_out_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_bits_qx_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_bits_swiglu_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_bits_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_gate_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_geglu_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_int6_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_qx_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_swiglu_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_u8_ws_cuda` - MFQ CUDA Python binding.
- `nint_gemv_packed_ws_cuda` - MFQ CUDA Python binding.
- `nint_mmq_cuda` - MFQ CUDA Python binding.
- `nint_mmq_f16_packed_cuda` - MFQ CUDA Python binding.
- `nint_mmq_gs24_f16_nint3_cuda` - MFQ CUDA Python binding.
- `nint_mmq_gs24_f16_nint4_cuda` - MFQ CUDA Python binding.
- `nint_mmq_gs24_f16_nint6_split4_ws_cuda` - MFQ CUDA Python binding.
- `nint_mmq_gs24_group32_ws_cuda` - MFQ CUDA Python binding.
- `nint_mmq_packed_exec_ws_cuda` - MFQ CUDA Python binding.
- `nint_mmq_packed_u8_ws_cuda` - MFQ CUDA Python binding.
- `nint_mmq_packed_ws_cuda` - MFQ CUDA Python binding.
- `nint_moe_grouped_matmul_hetero_f16_cuda` - MFQ CUDA Python binding.
- `nint_moe_grouped_matmul_hetero_qx_cuda` - MFQ CUDA Python binding.
- `nint_moe_grouped_matmul_pool_ws_cuda` - MFQ CUDA Python binding.
- `nint_moe_quantize_24_28_ws_cuda` - MFQ CUDA Python binding.
- `nint_moe_quantize_geglu_input_ws_cuda` - MFQ CUDA Python binding.
- `nint_moe_quantize_input_ws_cuda` - MFQ CUDA Python binding.
- `nint_moe_quantize_swiglu_24_28_ws_cuda` - MFQ CUDA Python binding.
- `nint_moe_quantize_swiglu_input_ws_cuda` - MFQ CUDA Python binding.
- `nvq2_gemv_swiglu_vec4_ordered_ws_cuda` - MFQ CUDA Python binding.
- `nvq_dequant_cuda` - MFQ CUDA Python binding.
- `nvq_embedding_lookup_cuda` - MFQ CUDA Python binding.
- `nvq_ffn_swiglu_quant_ws_cuda` - MFQ CUDA Python binding.
- `nvq_gemm_f16_cuda` - MFQ CUDA Python binding.
- `nvq_gemv_batch_vec8_ws_cuda` - MFQ CUDA Python binding.
- `nvq_gemv_gate_ws_cuda` - MFQ CUDA Python binding.
- `nvq_gemv_m1_vec8_ws_cuda` - MFQ CUDA Python binding.
- `nvq_gemv_multi2_ws_cuda` - MFQ CUDA Python binding.
- `nvq_gemv_qx_ws_cuda` - MFQ CUDA Python binding.
- `nvq_gemv_swiglu_ws_cuda` - MFQ CUDA Python binding.
- `nvq_gemv_ws_cuda` - MFQ CUDA Python binding.
- `nvq_mmq_gate_ws_cuda` - MFQ CUDA Python binding.
- `nvq_mmq_ws_cuda` - MFQ CUDA Python binding.
- `nvq_moe_grouped_matmul_pool_ws_cuda` - MFQ CUDA Python binding.
- `rms_norm_cuda` - MFQ CUDA Python binding.
- `rms_norm_f16_cuda` - MFQ CUDA Python binding.
- `rms_norm_offset_cuda` - MFQ CUDA Python binding.
- `rope_cuda` - MFQ CUDA Python binding.
- `rope_ext_cuda` - MFQ CUDA Python binding.
- `rope_table_cuda` - MFQ CUDA Python binding.
- `sample_apply_penalties_cuda` - MFQ CUDA Python binding.
- `sample_greedy_cuda` - MFQ CUDA Python binding.
- `sample_softmax_cuda` - MFQ CUDA Python binding.
- `sample_token_counts_add_cuda` - MFQ CUDA Python binding.
- `sample_top_k_top_p_cuda` - MFQ CUDA Python binding.
- `silu_mul_cuda` - MFQ CUDA Python binding.
- `ssm_conv_silu_cuda` - MFQ CUDA Python binding.
- `ssm_conv_silu_decode_cuda` - MFQ CUDA Python binding.
<!-- MFQ_NATIVE_BINDINGS_END -->

### 9.2 Native runtime 直接调用、未导出到 Python 的 CUDA 入口（43）

这些入口直接链接进 `mfq-decode`。它们不出现在 `mfq_cuda.cpp` 的
Python binding 中，仍属于生产运行时算子。

<!-- MFQ_RUNTIME_ONLY_BINDINGS_BEGIN -->
- `attention_cache_decode_split_cuda` — decode Attention split-K partial/reduce。
- `attention_cache_swa_planned_cuda` — 固定 planned length 的 SWA decode。
- `attention_dsv4_sparse_cuda` — DSV4 sparse/compressed Attention。
- `attention_glm_mla_sparse_cuda` — GLM DSA sparse MLA。
- `attention_glm_mla576_cached_cuda` — GLM MLA576 cached prefill。
- `attention_glm_mla576_cuda` — GLM MLA576 dense prefill。
- `attention_glm_mla576_decode_cuda` — GLM MLA576 decode。
- `attention_llama_flash256_cuda` — llama-style D256 prefill Flash Attention。
- `attention_llama_flash256_decode_cuda` — llama-style D256 decode。
- `attention_llama_flash256_swa_cuda` — llama-style D256 SWA prefill。
- `attention_llama_flash256_swa_decode_cuda` — llama-style D256 SWA decode。
- `attention_llama_flash512_cuda` — llama-style D512 prefill。
- `attention_llama_flash512_decode_cuda` — llama-style D512 decode。
- `dsv4_build_decode_plan_cuda` — DSV4 decode sparse plan。
- `dsv4_build_prefill_plan_cuda` — DSV4 prefill sparse plan。
- `dsv4_compress_cuda` — DSV4 KV/gate compressor。
- `dsv4_decode_pool_update_cuda` — DSV4 incremental compressed pool update。
- `dsv4_fp4_sim_cuda` — DSV4 FP4 cache 数值模拟。
- `dsv4_hc_post_cuda` — DSV4 Hyper-Connection post merge。
- `dsv4_hc_pre_cuda` — DSV4 Hyper-Connection pre。
- `dsv4_indexer_scores_cuda` — DSV4 indexer score。
- `dsv4_topk512_cuda` — DSV4 固定 512 top-k。
- `gdn_inplace_tiled_cuda` — tiled Q/K heads GDN。
- `gdn_inplace_transposed_tiled_cuda` — tiled heads + transposed-state GDN，Qwen3.6 默认。
- `glm_dsa_cache_write_cuda` — GLM index/latent cache write。
- `glm_dsa_indexer_layer_norm_cuda` — GLM indexer layer norm。
- `glm_dsa_indexer_scores_cuda` — GLM indexer prefill scores。
- `glm_dsa_indexer_scores_decode_cuda` — GLM indexer decode scores。
- `glm_interleaved_rope_cuda` — GLM interleaved RoPE。
- `linear_conv_qkv_prefill_cuda` — Qwen linear Attention prefill conv + Q/K L2。
- `linear_gate_beta_cuda` — Qwen linear Attention alpha/beta transform。
- `nint_cublas_gemm_nt_f32acc_cuda` — NINT dequant 后 FP32-accumulate cuBLAS。
- `nint_gemv_packed_u8_groupwise_ws_cuda` — DSV4 groupwise NINT8 projection。
- `nint_moe_grouped_matmul_hetero_f16_slice_cuda` — heterogeneous MoE output-row slice。
- `nint_moe_grouped_matmul_hetero_glu_qx_cuda` — heterogeneous MoE GLU 后预量化输入。
- `nint_moe_quantize_geglu_24_28_ws_cuda` — GeGLU gs24/28 双量化。
- `nint_moe_quantize_swiglu_clamped_input_ws_cuda` — clamped SwiGLU activation quantize。
- `nint_moe_set_small_mmq_cuda` — 设置小 M routed MMQ 策略。
- `nint8_zero_dequant_cuda` — NINT8-0 full dequant。
- `nint8_zero_embedding_lookup_cuda` — NINT8-0 embedding 行解码。
- `nint8_zero_gemv_ws_cuda` — NINT8-0 GEMV。
- `nint8_zero_mmq_ws_cuda` — NINT8-0 MMQ。
- `nint8_zero_moe_grouped_matmul_pool_ws_cuda` — NINT8-0 cohort grouped MoE。
<!-- MFQ_RUNTIME_ONLY_BINDINGS_END -->

### 9.3 内置 TPQ CUDA 扩展（62）

<!-- MFQ_TPQ_BINDINGS_BEGIN -->
- `attention_residual_bf16` - bundled TPQ/CCCP CUDA binding.
- `bf16_gemv_out` - bundled TPQ/CCCP CUDA binding.
- `block_fp8_gemv_f32` - bundled TPQ/CCCP CUDA binding.
- `dsv4_attn_decode` - bundled TPQ/CCCP CUDA binding.
- `dsv4_hc_post` - bundled TPQ/CCCP CUDA binding.
- `dsv4_hc_pre` - bundled TPQ/CCCP CUDA binding.
- `dsv4_hc_pre_norm` - bundled TPQ/CCCP CUDA binding.
- `dsv4_route_post` - bundled TPQ/CCCP CUDA binding.
- `expert_dispatch_pack` - bundled TPQ/CCCP CUDA binding.
- `flashinfer_mla_batch1_plan` - bundled TPQ/CCCP CUDA binding.
- `gated_activation_bf16` - bundled TPQ/CCCP CUDA binding.
- `glm_ep_reduce_residual` - bundled TPQ/CCCP CUDA binding.
- `glm_latent_kv_decode_prepare` - bundled TPQ/CCCP CUDA binding.
- `glm_merge_scores` - bundled TPQ/CCCP CUDA binding.
- `glm_mla_bmm_decode` - bundled TPQ/CCCP CUDA binding.
- `glm_moe_residual_add` - bundled TPQ/CCCP CUDA binding.
- `glm_norm_qkv_int4` - bundled TPQ/CCCP CUDA binding.
- `glm_residual_norm_router` - bundled TPQ/CCCP CUDA binding.
- `glm_residual_norm_router_norm_out` - bundled TPQ/CCCP CUDA binding.
- `glm_rope_qk` - bundled TPQ/CCCP CUDA binding.
- `glm_route` - bundled TPQ/CCCP CUDA binding.
- `glm_route_out` - bundled TPQ/CCCP CUDA binding.
- `hadamard_bf16` - bundled TPQ/CCCP CUDA binding.
- `hc_sinkhorn` - bundled TPQ/CCCP CUDA binding.
- `int4_embedding_lookup` - bundled TPQ/CCCP CUDA binding.
- `int4_embedding_lookup_device_row` - bundled TPQ/CCCP CUDA binding.
- `int4_gemv_packed_f32` - bundled TPQ/CCCP CUDA binding.
- `int4_glm_qb_split` - bundled TPQ/CCCP CUDA binding.
- `int4_swiglu_packed_f32` - bundled TPQ/CCCP CUDA binding.
- `kimi_gated_rmsnorm` - bundled TPQ/CCCP CUDA binding.
- `kimi_kda_recurrent` - bundled TPQ/CCCP CUDA binding.
- `kimi_moe_packed` - bundled TPQ/CCCP CUDA binding.
- `kimi_short_conv3` - bundled TPQ/CCCP CUDA binding.
- `latent_mla_attention_decode` - bundled TPQ/CCCP CUDA binding.
- `latent_mla_attention_scores` - bundled TPQ/CCCP CUDA binding.
- `launch_cuda_graphs` - bundled TPQ/CCCP CUDA binding.
- `launch_cuda_graphs_reduce` - bundled TPQ/CCCP CUDA binding.
- `launch_cuda_graphs_reduce_norm_router` - bundled TPQ/CCCP CUDA binding.
- `linear_sigmoid_route_out` - bundled TPQ/CCCP CUDA binding.
- `moe_mlp_routed_codegemm` - bundled TPQ/CCCP CUDA binding.
- `moe_mlp_routed_slots` - bundled TPQ/CCCP CUDA binding.
- `moe_mlp_routed_vv` - bundled TPQ/CCCP CUDA binding.
- `moe_mlp_slots` - bundled TPQ/CCCP CUDA binding.
- `pack_vq_tensor_shard_codegemm` - bundled TPQ/CCCP CUDA binding.
- `packed_moe_topk` - bundled TPQ/CCCP CUDA binding.
- `paged_gather_bf16` - bundled TPQ/CCCP CUDA binding.
- `residual_add3` - bundled TPQ/CCCP CUDA binding.
- `rmsnorm` - bundled TPQ/CCCP CUDA binding.
- `rmsnorm_bf16` - bundled TPQ/CCCP CUDA binding.
- `rope1` - bundled TPQ/CCCP CUDA binding.
- `sigmoid_route` - bundled TPQ/CCCP CUDA binding.
- `sigmoid_route_out` - bundled TPQ/CCCP CUDA binding.
- `tp_all_rank_reduce` - bundled TPQ/CCCP CUDA binding.
- `tp_attention_peer_dispatch` - bundled TPQ/CCCP CUDA binding.
- `tp_attention_source_pack` - bundled TPQ/CCCP CUDA binding.
- `tp_hidden_add_batch` - bundled TPQ/CCCP CUDA binding.
- `tp_hidden_residual_mix_batch` - bundled TPQ/CCCP CUDA binding.
- `tp_hidden_rmsnorm_batch` - bundled TPQ/CCCP CUDA binding.
- `tp_peer_copy` - bundled TPQ/CCCP CUDA binding.
- `unpack_vq_codegemm` - bundled TPQ/CCCP CUDA binding.
- `vq_gemv` - bundled TPQ/CCCP CUDA binding.
- `vq_gemv_slots_out` - bundled TPQ/CCCP CUDA binding.
<!-- MFQ_TPQ_BINDINGS_END -->

| 入口 | dtype/shape | launch geometry | 限制与工作区 |
|---|---|---|---|
| `vq_gemv` | FP32 `x[N\|1,C]`、`idx[N,R,B]`、`cb[N\|1,K,D]` | `block=(32,8)`，`grid=(ceil(R/8),N)` | 每 warp 一行；dynamic shared=`C*4`；u8/u16 index |
| `vq_gemv_slots_out` | BF16，最多 8 个 slot expert | `block=(32,8)`，`grid=(ceil(R/8),N)` | 每 expert 的 B/D 可不同；shared=`C*2`；写 caller workspace |
| `moe_mlp_slots` | BF16 GU/DN，FP32 route weights | 两次 slot GEMV + 256-thread SwiGLU + 256-thread weighted sum | 共 4 次 launch；`N<=8`；FP32 激活/路由累加后写 BF16 |
| `hc_sinkhorn` | FP32 `[N,24]` | `block=128`，`grid=ceil(N/128)` | 每 thread 一行 4x4 Sinkhorn，矩阵留在寄存器 |
| `rmsnorm` | FP32 rows | `grid=N`，`block=256` | FP32 reduction |
| `rope1` | FP32 decode、交错 RoPE | `grid=N`，`block=64` | 所有行共享单相位 cos/sin |
| `dsv4_attn_decode` | FP32 `q[1,H,D]`、window/compressed KV | `grid=H`，`block=128` | 单 block/head；`W+C<=4096`；shared=`(2D+W+C+4)*4` |
| `dsv4_hc_pre` | FP32 state，FP32/BF16 fn | `grid=N`，`block=(32,8)` | 每 token 一块；输出 FP32 reduced/post/comb |
| `dsv4_hc_pre_norm` | BF16 state，FP32/BF16 fn/norm | BF16 fn+norm 且 `D>=48`：`N*24` 个 256-thread mix blocks，再 `N` 个 256-thread finish blocks；其余 `grid=N, block=(32,8)` | 前者增加一次全局中间写，换取 24 行并行 |
| `dsv4_hc_post` | BF16 residual/post/comb，out 为 FP32/BF16 | `grid=N*4`，`block=256` | 每 block 一个输出 channel；FP32 accumulation，BF16 output |
| `dsv4_route_post` | FP32 scores/bias `[1,E]`、bool mask | 单 block；`E<256` 用 128 threads，否则 256 | `E<=1024`、`top_k<=16`；shared=`E*4` |
| `paged_gather_bf16` | BF16 paged KV、int64 indices | `block=256`，`grid=ceil(items*dim/256)` | 当前 fused wrapper 只接 B=1 BF16 |
| `hadamard_bf16` | BF16 `[rows,width]` | `grid=rows`，`block=width` | width 为 2 的幂且 `<=256`；shared=`width*4` |

TPQ 扩展以 `torch.utils.cpp_extension.load` lazy build。`TPQ_FUSED=0` 或编译失败时，
Python 图会进入 torch fallback；这条路径可做数值 reference，不能直接用于生产
性能结论。slot MLP 只服务 `B*T=1`、全部选中 expert 都是兼容 VQWeight 的情形；
混合 signature 会先分组，无法使用 slot kernel 时进入 batched LUT 实现。

### 9.4 离线量化 CUDA 扩展（8）

这组入口由 `mfq/quantize/cuda/_ext.py` 单独编译，不进入推理 binary。

<!-- MFQ_QUANT_BINDINGS_BEGIN -->
- `nvq1_l_assign` — NVQ1-L 固定 anchor 的精确 group assignment。
- `nvq_search` — NVQ 浮点 group-scale 与 code assignment 联合搜索。
- `nvq_reassign` — 固定 scale 后重新选择 NVQ code。
- `nepq0_s_assign` — NEPQ0-S 256-bank assignment 与 anchor refit。
- `nvq2j_assign` — NVQ2J state/bank/code assignment 与 anchor refit。
- `nvq2j_search_banks` — NVQ2J 多 bank 训练期候选搜索。
- `nint_make_qkx3` — NINT imatrix-weighted affine group search。
- `nint_make_qp` — NINT weighted neuron scale/min quantization。
<!-- MFQ_QUANT_BINDINGS_END -->

| 算子 | geometry | 工作单位 |
|---|---|---|
| NVQ1-L assignment | 每 group 一个 256-thread block，dynamic shared codebook | gs24 |
| NVQ2/NVQ3 search/reassign | 每 group 一个 256-thread block | E8 3×8D 或 D4 6×4D |
| NVQ2J assignment | 每 group 96 threads；每 row 256-thread anchor refit | 16 states、4 banks、3×8D |
| NVQ2J bank search | assignment、partial reduce、finalize 三阶段 | 多 bank 训练候选与全局误差归约 |
| NEPQ0-S assignment | 每 96-weight supergroup 一个 256-thread block；每 row refit | 256 banks、4 groups/supergroup |
| NINT qkx3 | 8 warps/block，每 warp 一个 group | weighted affine search |
| NINT qp | 每 row 一个 warp | 顶层 scale/min quantization |

## 10. CUDA 文件与职责

下面统计源码中的 `__global__` 定义。模板、split/reduce、不同 dtype/tile 的内部
kernel 都计入，共 270 个定义；它们会组合成第 9 节的 235 条可调用入口。

<!-- MFQ_CUDA_SOURCE_TABLE_BEGIN -->
| Source | `__global__` count | Responsibility |
|---|---:|---|
| `mfq/_vendor/tpq/csrc/vq_gemv.cu` | 66 | VQ GEMV/slot MLP、HC、DSV4 decode、paged gather |
| `mfq/kernels/cuda/acc.cu` | 5 | residual、residual+norm、Gemma4 merge |
| `mfq/kernels/cuda/activation.cu` | 4 | SiLU/GELU gate、linear alpha/beta |
| `mfq/kernels/cuda/attention.cu` | 9 | 通用 full/SWA/cache attention |
| `mfq/kernels/cuda/attention_llama.cu` | 4 | llama-style Flash256/512 prefill/decode |
| `mfq/kernels/cuda/deepseek_v4_attention.cu` | 8 | DSV4 plan/cache/attention/pool |
| `mfq/kernels/cuda/deepseek_v4_hc.cu` | 2 | HC pre/post |
| `mfq/kernels/cuda/embedding.cu` | 6 | dense/NINT/NINT8 embedding |
| `mfq/kernels/cuda/gated_delta_net.cu` | 3 | GDN shared/column/warp-column |
| `mfq/kernels/cuda/glm_dsa.cu` | 5 | GLM sparse attention/indexer |
| `mfq/kernels/cuda/kv_cache.cu` | 3 | linear/ring KV write |
| `mfq/kernels/cuda/moe.cu` | 33 | route、activation quantize、NINTM grouped、reduce |
| `mfq/kernels/cuda/nepq.cu` | 1 | signed Hadamard |
| `mfq/kernels/cuda/nint_matmul.cu` | 63 | NINT GEMV/MMQ/dequant/FFN/LM-head |
| `mfq/kernels/cuda/norm.cu` | 3 | RMS/L2 norm |
| `mfq/kernels/cuda/nvq_matmul.cu` | 24 | NVQ/NPQ/NEPQ GEMV/MMQ/dequant/grouped |
| `mfq/kernels/cuda/rope.cu` | 2 | RoPE variants |
| `mfq/kernels/cuda/sampling.cu` | 8 | argmax、softmax、top-k/top-p、penalties |
| `mfq/kernels/cuda/ssm_conv.cu` | 7 | causal depthwise conv 与 QKV decode fusion |
| `mfq/quantize/cuda/nepq0_s_assign.cu` | 2 | NEPQ0-S assignment/refit |
| `mfq/quantize/cuda/nint_quant.cu` | 2 | NINT qkx3/qp search |
| `mfq/quantize/cuda/nvq2j_assign.cu` | 7 | NVQ2J assignment/refit 与多 bank 搜索 |
| `mfq/quantize/cuda/nvq_quant_assign.cu` | 3 | NVQ1/NVQ search/reassign |
<!-- MFQ_CUDA_SOURCE_TABLE_END -->

## 11. 校准与码率分配

### 11.1 端到端流程

```text
固定 token corpus
 -> 收集 train/validation E[x²] 与 row Fisher
 -> 每 tensor/profile 做真实量化和 surrogate loss
 -> 同 runtime-compatible group 建候选
 -> 固定预算分配
 -> layerwise replay
 -> 可选 soft end-to-end KL + dual rate constraint
 -> 离散化并在 validation 上接受/拒绝
 -> scheme
```

surrogate 的逐行目标为：

```text
loss_row = row_fisher[row] *
           sum_i input_second_moment[i] * (W[row,i]-Wq[row,i])²
```

它同时使用输入通道二阶矩和输出行 Fisher。最终选择仍需用真实模型图的
hidden trace 或 terminal KL 验证。

| 分配器 | 决策变量 | 约束/目标 | 位置 |
|---|---|---|---|
| basic allocator | 每 precision group 一个候选 | 总 storage bits 内最小 surrogate loss | `allocator.py` |
| layer strategies | 同层 runtime-compatible tensor 联动 | layer/global/compensated budget | `evaluator.py`, `refinement.py` |
| soft refinement | 每 group 的 precision logits | KL + dual×rate violation，温度退火 | `soft_refinement.py` |
| discrete refinement | 单组/双组 profile 替换 | 固定预算内验证真实 metric | `rate_distortion.py` |
| REAP EW | 每 expert profile | exposure-weighted gate/up+down NMSE | `reap_expertwise.py` |
| coupled MoE | 同 expert 的 gate/up/down 联动 | 保持 expert 级配方和预算 | `moe_soft_refinement.py` |
| CCCP tier | x/w/v/vv | 固定 tier 或 score coverage | `calibration/cccp.py` |

REAP allocation 的 expert loss：

```text
normalized_exposure *
0.5 * (gate_up_nmse + down_nmse)
```

因此同一 expert 的 gate/up 与 down 默认作为一个预算决策。若实验需要拆分，
必须另设受控方案，不能悄然改变现有 EW 定义。

### 11.2 数据隔离与复现字段

每个校准或评测工件至少保存：

| 类别 | 字段 |
|---|---|
| 模型 | source path/revision、index SHA-256、architecture |
| 数据 | dataset/revision、token SHA-256、split、domain、render mode |
| tokenization | tokenizer identity、chat template/render mode、BOS/EOS 处理 |
| 采样 | seed、tokens、sequence length、chunk IDs |
| 量化 | format/spec、imatrix identity、codebook identity、backend |
| 目标 | surrogate/KLD 定义、scored positions、budget bits |
| 运行 | device、dtype、row/token chunk、版本/commit |

train 数据用于统计、码本和 soft optimization；validation 数据用于候选排序审计、
离散方案接受和最终报告。禁止用 validation metric 反复调整同一方案后仍把它报告为
未见验证集。

## 12. 构建、修改与测试

### 12.1 两套 CUDA 构建

| 目标 | 构建方式 | 源文件 |
|---|---|---|
| Python inference extension | 首次 `ext()` 时 `torch.utils.cpp_extension.load` | 15 个 runtime `.cu/.cpp` |
| offline quant extension | 首次 quant CUDA 调用时单独 lazy build | 4 个 assignment `.cu` + binding |
| native `mfq-decode` | CMake + LibTorch + CUDA + llama shared library | runtime C++、server、18 个 CUDA source |
| TPQ extension | TPQ loader lazy build | `_vendor/tpq/csrc/vq_gemv.cu` |

Windows native build：

```powershell
cmake -S cpp_runtime -B build/cpp_runtime `
  -DMFQ_LLAMA_BUILD_DIR=<llama-build> `
  -DMFQ_CUDA_ARCHITECTURES=89
cmake --build build/cpp_runtime --target mfq-decode -j 8
```

当前 CMake 默认 architecture 是 86。部署到 Ada/Hopper 前必须显式设置目标
architecture，避免只验证 PTX/JIT 或编错目标。

### 12.2 修改影响矩阵

| 改动 | 必须同步检查 |
|---|---|
| dtype/bitstream | Python pack/unpack、native loader、mmap、size estimator、roundtrip tests |
| NINTM cohort | family label、artifact/rotation key、expert map、pool loader、heterogeneous dispatch |
| quantizer | CPU/CUDA parity、imatrix path、fused/non-fused parity、artifact identity |
| dense kernel | Python wrapper、native declaration/dispatch、M/N/K sweep、fused FFN |
| MoE kernel | route map、shared/routed input、mixed cohorts、T=1/2/4/8/16/512 |
| Attention/state | prefill/decode、cache reset、B>1、variable seq_len、graph capture |
| KLD evaluator | slow/optimized、token shift、chunk protocol、UD reference |
| offload/cache | cold/warm 分离、prefetch event、eviction、resident budget、数值 parity |
| TP | slice alignment、gate/up paired order、partial reduce、单卡 parity |

最小验证集合：

```powershell
python -m pytest -q -p no:cacheprovider tests/test_operator_reference.py
python -m pytest -q -p no:cacheprovider tests/test_formats
python -m pytest -q -p no:cacheprovider tests/test_kernels_ops.py
python -m pytest -q -p no:cacheprovider tests/test_cpp_runtime_kl_evaluator_source.py
```

CUDA 改动还需运行对应的 `test_*_kernels_cuda.py`；native 改动需重新构建
`mfq-decode`。source-level test 只能确认分发契约存在，不能代替 GPU 数值测试。

## 13. 运行时开关

这些开关会改变数值路径或性能结果。默认值由对应源码决定；表中只列开发时常用项。

| 类别 | 环境变量 | 默认/作用 |
|---|---|---|
| overlay | `MFQ_TENSOR_OVERLAY`, `MFQ_EXPERT_OVERLAY` | 完整 tensor 或 NINTMD expert delta |
| Attention | `MFQ_LLAMA_FLASH256`, `MFQ_LLAMA_FLASH_DECODE` | 控制 llama-style 专用路径 |
| Attention | `MFQ_ATTENTION_SPLITK`, `MFQ_ATTENTION_DECODE_SPLITK` | 控制 generic/specialized split-K |
| Attention | `MFQ_ATTENTION_DECODE_ATEN`, `MFQ_KV_CACHE_WRITE_ATEN` | 诊断用 ATen 路径 |
| GDN | `MFQ_GDN_WARP`, `MFQ_GDN_COLUMN`, `MFQ_GDN_TRANSPOSED_STATE` | warp-column 默认开启；column 仅小 T |
| NINT | `MFQ_NINT3_PREFILL_PATH` | `group32`/`f16`/`dequant` |
| NINT | `MFQ_NINT4_BATCH_GEMV_MAX_M` | 默认 16 |
| NINT | `MFQ_NINT_HI_GEMV_MAX_M` | 默认 8 |
| NINT | `MFQ_NINT8_PREFILL_MMQ` | 强制 NINT8 packed MMQ |
| MoE | `MFQ_DISABLE_MOE_HETERO` | 禁用 mixed-cohort 单次 dispatch |
| MoE | `MFQ_DISABLE_MOE_PREFILL_MMA` | 禁用 heterogeneous FP16 prefill MMA |
| MoE | `MFQ_MOE_PREFILL_MMA_MIN_TOKENS` | 默认 256 |
| MoE | `MFQ_MOE_PREFILL_MMA_BLOCKS` | 默认最多 4096 blocks |
| MoE | `MFQ_MOE_PREFILL_MMA_BM` | 强制 16/32/64 |
| MoE | `MFQ_MOE_TOKEN_WARPS` | 默认 16，可设 8/16/32 |
| MoE | `MFQ_MOE_SMALL_MMQ` | 覆盖 gs24/28 small-M group32 MMA |
| MoE | `MFQ_DISABLE_MOE_PERSISTENT` | 禁用小 T persistent tile |
| NVQ MoE | `MFQ_NVQ_MOE_WARPS`, `MFQ_NVQ_MOE_ROWS_PER_BLOCK` | 小 T geometry override |
| NVQ MoE | `MFQ_NVQ_MOE_EXACT_REDUCTION` | ordered/exact reduction 诊断 |
| fusion | `MFQ_DISABLE_NVQ_FUSION`, `MFQ_DISABLE_FFN_*_FUSION` | 关闭对应融合 |
| graph | `MFQ_CUDA_GRAPH`, `MFQ_SERVER_CUDA_GRAPH` | decode/server graph |
| profiling | `MFQ_REPORT_CUDA_MEMORY`, `MFQ_PROFILE_SCOPES` | allocator/范围统计 |
| offload | `MFQ_ENABLE_MOE_HISTORY_PREFETCH` | 上一 token route 预取，默认关闭 |

TPQ/CCCP 开关：

| 类别 | 环境变量 | 默认/作用 |
|---|---|---|
| compute | `TPQ_COMPUTE_DTYPE` | `auto`；SM80+ BF16、SM70 FP16、CPU/其他 FP32 |
| dense | `TPQ_DENSE_BF16` | `none`；可选 `attention,compressor,embed,head,hyper,indexer,norm,shared` 或 `all` |
| dense | `TPQ_INT4_HALF` | `0`；开启 INT4 dense 的 FP16 compute，仅用于对照 |
| KV | `TPQ_LATENT_KV` | `1`；GLM 保存 latent KV，关闭后保存逐头完整 K/V |
| KV | `TPQ_SINGLE_TOKEN_ATTN_FAST` | `1`；DSV4 单 token fused Attention |
| KV | `TPQ_DIRECT_KV_PREFIX` | `1`；压缩 KV 尚在首个 page 时直接取 contiguous prefix |
| KV | `TPQ_PAGED_KV_FUSED` | `1`；BF16 paged gather CUDA kernel |
| KV | `TPQ_PAGED_KV_STRICT` | `0`；设 1 强制 Python strict gather/reference |
| indexer | `TPQ_INDEXER_HADAMARD_FUSED` | `1`；BF16 Hadamard CUDA kernel |
| expert compute | `TPQ_GROUPED` | `1`；decode grouped VQ MLP |
| expert compute | `TPQ_SLOT_VQ` | `1`；固定 arena slot 的 4-launch VQ MLP |
| expert compute | `TPQ_FUSED` | `1`；加载 TPQ CUDA 扩展 |
| expert cache | `TPQ_LOAD_WORKERS` | `12`；NVMe expert 并行读取线程 |
| expert cache | `TPQ_READ_BUF_MB` | `2`；每个线程局部文件句柄缓冲 |
| expert cache | `TPQ_PROFILE_JSON` | 模型目录 `profile.json`；也可指定外部热度排名 |
| expert cache | `TPQ_PIN_GB` | `0`；按热度排名永久驻 RAM 的预算 |
| expert cache | `TPQ_FULL_RESIDENT` | `1`；容量允许时尝试全部 expert 常驻 RAM |
| expert cache | `TPQ_RESIDENT_RESERVE_GB` | `3`；全量 RAM 常驻判定的系统内存余量 |
| expert cache | `TPQ_HOST_PIN_GB` | `auto`；page-locked 常驻 expert 上限，`0` 关闭 |
| expert cache | TPQ_GPU_FULL_RESIDENT | `auto`；arena 足够时全部 expert 常驻 GPU |
| expert cache | `TPQ_VRAM_RUNTIME_GB` | DSV4 默认 1.5、GLM 默认 3.0；dense/cache 之外的运行时余量 |
| expert cache | `TPQ_PREFETCH` | `auto`；按上一 token 路由预测下一 token |
| expert cache | `TPQ_PREFETCH_STAGE` | `1`；预取同时做 RAM→VRAM DMA，`0` 只预读磁盘 |
| expert cache | `TPQ_STAGE_SYNC` | `0`；设 1 使用同步 `.to()` 诊断 staging |
| expert cache | `TPQ_STAGE_VERIFY` | `0`；设非零逐对校验 DMA 结果 |
| VRAM | `TPQ_VRAM_RESERVE_GB` | `1.25`；初始化时 per-process allocator 硬上限余量 |
| VRAM | `TPQ_VRAM_WATCH` | `1`；动态 cache budget 监视 |
| VRAM | `TPQ_VRAM_WATCH_LOW_GB` | `0.8`；触发缩减 budget 的空闲显存阈值 |
| VRAM | `TPQ_VRAM_WATCH_HIGH_GB` | `3.0`；触发恢复 budget 的空闲显存阈值 |
| VRAM | `TPQ_VRAM_WATCH_SEC` | `3`；监视周期秒数 |
| DSpark | `TPQ_DSPARK_EXPERIMENTAL` | 未设置；DSV4 speculative 路径需显式启用 |
| DSpark | `TPQ_DSPARK_GB` | `1.5`；packed expert RAM LRU |
| DSpark | `TPQ_DSPARK_VRAM_GB` | `2.75`；运行 DSpark 前预留的 VRAM |

kernel 内还有针对单一 shape 的调优开关，如
`MFQ_NINT*_GS*_GROUP/SPLIT/WARPS`。它们只能用于 A/B 和交叉点重测，
不能作为未记录的生产依赖。

<!-- MFQ_TPQ_ENV_BEGIN -->
### TPQ/CCCP complete environment-switch inventory

- `TPQ_ACCUMULATE_MOE_RANK`
- `TPQ_ALL_RANK_ARGUMENTS`
- `TPQ_API_KEY`
- `TPQ_ATTENTION_GRAPH`
- `TPQ_ATTENTION_TENSOR_WORKSPACES`
- `TPQ_ATTENTION_TP`
- `TPQ_BASE_URL`
- `TPQ_CODEGEMM_DISPATCH_GRAPH`
- `TPQ_CODEGEMM_GRAPH`
- `TPQ_CODEGEMM_VQ`
- `TPQ_COMPUTE_DTYPE`
- `TPQ_CPU_ATTN_MANY`
- `TPQ_CPU_BF16`
- `TPQ_CPU_DN_BLOCK`
- `TPQ_CPU_EXPAND_BF16`
- `TPQ_CPU_FUSED`
- `TPQ_CPU_MOE_PROFILE`
- `TPQ_CPU_NUMA`
- `TPQ_CPU_PACKED`
- `TPQ_CPU_PACKED_DIRECT`
- `TPQ_CPU_PACKED_MOE`
- `TPQ_CPU_PACKED_PROFILE`
- `TPQ_CPU_QKV_POST`
- `TPQ_CPU_THREADS`
- `TPQ_CPU_VQ_INT8`
- `TPQ_CPU_W4A8`
- `TPQ_CPU_W4ABF16`
- `TPQ_DECODE_WORKSPACES`
- `TPQ_DENSE_BF16`
- `TPQ_DENSE_TP`
- `TPQ_DIRECT_KV_PREFIX`
- `TPQ_DOWN_REDUCE_ROWS`
- `TPQ_DSPARK_EXPERIMENTAL`
- `TPQ_DSPARK_GB`
- `TPQ_DSPARK_VRAM_GB`
- `TPQ_EP_DEVICE_ROUTE`
- `TPQ_EP_DIRECT_RETURN`
- `TPQ_EP_FUSED_DISPATCH`
- `TPQ_EP_LAYOUT`
- `TPQ_EP_OVERLAP_SHARED`
- `TPQ_FIRST_DENSE_TP`
- `TPQ_FLASHINFER_BACKEND`
- `TPQ_FLASHINFER_GPU_PLAN`
- `TPQ_FLASHINFER_MLA`
- `TPQ_FP8_GEMV_FUSED`
- `TPQ_FULL_RESIDENT`
- `TPQ_FUSED`
- `TPQ_FUSED_DOWN_REDUCE`
- `TPQ_FUSED_MOE_POINTERS`
- `TPQ_GLM_CUBLAS_DECODE`
- `TPQ_GLM_CUBLAS_Q`
- `TPQ_GLM_CUBLAS_VALUE`
- `TPQ_GLM_DIRECT_BMM`
- `TPQ_GLM_EP_FINAL_FUSED`
- `TPQ_GLM_LATENT_PREP_FUSED`
- `TPQ_GLM_MOE_RESIDUAL_ADD`
- `TPQ_GLM_NORM_QKV_FUSED`
- `TPQ_GLM_QB_GROUP_VECTOR`
- `TPQ_GLM_QB_SPLIT`
- `TPQ_GLM_RESIDUAL_NORM_QKV`
- `TPQ_GLM_RESIDUAL_NORM_ROUTER`
- `TPQ_GLM_ROPE_FUSED`
- `TPQ_GLM_ROUTE_FUSED`
- `TPQ_GLM_SCORE_FUSED`
- `TPQ_GLM_SEQUENTIAL_PREFILL`
- `TPQ_GLM_SEQUENTIAL_PREFILL_MAX`
- `TPQ_GREEDY_DEVICE_WINDOW`
- `TPQ_GROUPED`
- `TPQ_HOST_PIN_GB`
- `TPQ_INDEXER_HADAMARD_FUSED`
- `TPQ_INT4_EMBEDDING_FUSED`
- `TPQ_INT4_GEMV_FUSED`
- `TPQ_INT4_GROUP_VECTOR`
- `TPQ_INT4_HALF`
- `TPQ_INT4_LM_HEAD_VECTOR`
- `TPQ_INT4_SWIGLU_FUSED`
- `TPQ_INT4_SWIGLU_GROUP_VECTOR`
- `TPQ_KIMI_ATTENTION_TP`
- `TPQ_KIMI_CUDA_EVENTS`
- `TPQ_KIMI_DENSE_TP`
- `TPQ_KIMI_LAYER_TIMING`
- `TPQ_KIMI_LAYER_TIMING_PRINT`
- `TPQ_KIMI_PACKED_HYBRID`
- `TPQ_KIMI_PROTECT_PREV`
- `TPQ_KIMI_RESIDENT_CODEBOOKS`
- `TPQ_KIMI_SLOT_MIX`
- `TPQ_KIMI_TP_GRAPH`
- `TPQ_KV_TRACE`
- `TPQ_LATENT_KV`
- `TPQ_LATENT_KV_INITIAL`
- `TPQ_LINEAR_ROUTE_FUSED`
- `TPQ_LM_HEAD_INT4`
- `TPQ_LM_HEAD_KEEP_F32`
- `TPQ_LOAD_WORKERS`
- `TPQ_MLA_TP`
- `TPQ_MODEL`
- `TPQ_MOE_OWNER_GRAPH`
- `TPQ_MOE_PARALLELISM`
- `TPQ_MOE_PRELUDE_TP`
- `TPQ_MOE_ROUTE_DOWN_TP`
- `TPQ_MOE_TP_GROUP`
- `TPQ_P12_CODES`
- `TPQ_P12_L2_WARPS`
- `TPQ_P12_ROWS_PER_BLOCK`
- `TPQ_P12_ROWS_PER_WARP`
- `TPQ_P12_SHARED`
- `TPQ_P12_SHARED_STRIDE`
- `TPQ_P12_WARPS`
- `TPQ_PAGED_KV_FUSED`
- `TPQ_PAGED_KV_STRICT`
- `TPQ_PAGED_LATENT_ATTENTION`
- `TPQ_PIN_GB`
- `TPQ_PREFETCH`
- `TPQ_PREFETCH_STAGE`
- `TPQ_PROFILE_JSON`
- `TPQ_PYTHON`
- `TPQ_RAM_MIRROR`
- `TPQ_RAM_RESERVE_GB`
- `TPQ_READ_BUF_MB`
- `TPQ_RESIDENT_RESERVE_GB`
- `TPQ_RESIDUAL_INVERSE_CACHE`
- `TPQ_RESIDUAL_MAX_ROWS`
- `TPQ_RESIDUAL_SINGLE_MAX_ROWS`
- `TPQ_RESIDUAL_STAGED_MAX_ROWS`
- `TPQ_RESIDUAL_THREADS`
- `TPQ_RESIDUAL_WARPS`
- `TPQ_RMSNORM_WORKSPACES`
- `TPQ_ROUTED_DOWN_ROW_TP`
- `TPQ_ROUTED_PROJECTION_TP`
- `TPQ_ROUTED_VECTOR_COPY`
- `TPQ_ROUTED_WARPS`
- `TPQ_ROUTER_TP`
- `TPQ_ROUTE_FUSED`
- `TPQ_ROUTE_HISTORY`
- `TPQ_ROUTE_RADIX`
- `TPQ_SHARED_MLP_TP`
- `TPQ_SINGLE_TOKEN_ATTN_FAST`
- `TPQ_SLOT_VQ`
- `TPQ_SMALL_OP_TP`
- `TPQ_SOURCE_COMMIT`
- `TPQ_SPEC`
- `TPQ_STAGE_SYNC`
- `TPQ_STAGE_VERIFY`
- `TPQ_STATIC_LM_OUTPUT`
- `TPQ_TP_DECODE_LAYER_PLAN`
- `TPQ_TP_DIRECT_INPUT`
- `TPQ_TP_EVENT_BARRIER`
- `TPQ_TP_FUSED_MOE_FINALIZE`
- `TPQ_TP_GRAPH`
- `TPQ_TP_GRAPH_SPIN`
- `TPQ_TP_HIDDEN`
- `TPQ_TP_HIDDEN_SERIAL_SHARED`
- `TPQ_TP_HIDDEN_STATE`
- `TPQ_TP_HIDDEN_SYNC_LAYER`
- `TPQ_TP_HIDDEN_TIMING`
- `TPQ_TP_HIDDEN_TRACE`
- `TPQ_TP_HIDDEN_TRACE_LAYER`
- `TPQ_TP_LAYER_GRAPH`
- `TPQ_TP_MOE_PLAN`
- `TPQ_TP_NO_OWNER`
- `TPQ_TP_PARALLEL_LAUNCH`
- `TPQ_TP_PERSISTENT_LAYER_PLAN`
- `TPQ_VQ_D4_SPECIALIZED`
- `TPQ_VRAM_RESERVE_GB`
- `TPQ_VRAM_RUNTIME_GB`
- `TPQ_VRAM_WATCH`
- `TPQ_VRAM_WATCH_HIGH_GB`
- `TPQ_VRAM_WATCH_LOW_GB`
- `TPQ_VRAM_WATCH_SEC`
- `TPQ_VV_CODES`
- `TPQ_VV_ROWS_PER_BLOCK`
- `TPQ_VV_ROWS_PER_WARP`
- `TPQ_VV_SHARED_STRIDE`
- `TPQ_VV_VECTOR`
- `TPQ_VV_WARPS_PER_BLOCK`
<!-- MFQ_TPQ_ENV_END -->

## 14. Profiler 与验证

Native `CudaProfiler` 使用 CUDA events 统计 GPU 时间，并记录调用次数和 wall time。
它不把 tensor 回读到 CPU。`--profile` 可用于 1–3 个 KLD chunk；
长全集 profiling 会产生不必要的 event 数量。

动态验证顺序：

1. 单算子 reference parity；
2. 单层 block trace；
3. 1 个 KLD chunk，与历史首块数值一致；
4. 3 个 KLD chunk，slow/optimized evaluator 一致；
5. 全集；
6. decode/prefill benchmark。

任何性能优化都需要同时记录：

| 项目 | 最低要求 |
|---|---|
| 数值 | max abs、RMSE/SNR；模型级 KLD/same-top |
| 性能 | kernel CUDA ms、整层 CUDA ms、端到端 wall |
| 内存 | allocated/reserved/进程显存，是否包含 cold load |
| 路径 | profiler label、调用数、fallback 数 |
| 对照 | 同 token、同 M、同激活 dtype、同累加与同步 |

## 15. 已知需要继续核对的项目

| 项目 | 当前状态 | 完成条件 |
|---|---|---|
| Qwen3.6 24Q/4KV SDPA 实际后端 | 静态确认落到 ATen SDPA | 记录 PyTorch backend selector 与 CUDA profile |
| common FP16 MMQ 的 M=512/2048 tile 效率 | geometry 已确认 | 与 llama.cpp Q8_1 MMQ 同矩阵 profile |
| NINT8-1 整数 MMQ 数值等价 | 尚未接入；现有模式只做 Q8_1 量化后 FP16 重建 | 接入与 llama.cpp 相同的 Q8_1 block 布局和整数 MMA，再做逐算子及 KLD 对齐 |
| REAP EW 的离线代理与整模 KLD | Qwen3.6-35B S4-L 的 routed-expert 普通权重 SNR 为 `22.1729 dB`，按当前 REAP exposure 加权后为 `25.8625 dB`；第 37/39 层相对 UD 的普通 SNR 分别低 `4.0337/4.9602 dB`，REAP 加权 SNR 却分别高 `3.3153/3.1784 dB`。Grouped NINTM 的 gate/up 执行附加误差约 `69.7 dB`，down 为逐元素零误差，因此现有证据不支持把主要差距归因于 NINTM 解包或 grouped kernel。REAP 原始 `reap` 已包含被选中专家的 router-weighted 输出 L2 norm，分配器也确实使用 `exposure = expertProbability * reap`；剩余偏差来自用“单个 expert 标量 × 权重 NMSE”近似实际输出误差：它没有描述输入激活协方差、量化误差方向、gate/up/down 各自的敏感度和误差在后续层的传播，并且逐层归一化会移除层间绝对尺度。 | 对每个 expert/profile 直接测量校准 token 上的 router-weighted 输出误差，分别记录 gate/up、down 和完整 expert 输出；与现有 scalar surrogate 做排序相关性及相同预算 KLD 对照 |
| GLM DSA 整模 | 源码已接入 | 独立完整数值与性能验证 |
| NEPQ 四格式整模 | kernel/loader 已接入 | 完整 KLD |
| TP 多 GPU | source/build/单测已有 | 真实多卡端到端数值与吞吐 |
| 通用异构 | cache/prewarm/overlap 已有 | 冷/热启动分离、不同 residency 曲线 |
| README scope | 与当前 TP/架构支持有旧描述 | 功能验证后同步 |
