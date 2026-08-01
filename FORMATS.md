# MFQ 量化格式总览

MFQ 当前有 **21 种基础编码**和 **4 种跨专家派生编码**，合计 25 种已实现的权重存储编码，
覆盖约 `0.84-8.30 bpw`：

- **NINT**：4-8 bit 神经元级两层仿射整数。
- **NVQ**：1.3-3 bit 神经元锚定向量量化。
- **NPQ**：约 1 bit 及以下的神经元锚定乘积量化。
- **NEPQ**：给基础 NPQ/NVQ 增加跨专家 256-bank、96-weight 选表。
- **TPQ**：共享学习码本的专家 Product-VQ，以及 dense INT4-G64。

`NINTM`、EW、梯度精度校准、imatrix、neuron gain 和 tensor-wise codebook 都不单独计为格式：
前者是混合专家容器，其余项目决定如何训练或分配已有编码。

表中 bpw 是代表性大矩阵的实际 payload 码率，包含 neuron anchor、group state 和 tensor 码本，
不含 MFQ 文件目录。精确值会随矩阵宽度、行数和位流取整略微变化。

## 21 种基础编码

| 格式 | 典型 bpw | 核心编码 | 主要定位 |
|---|---:|---|---|
| **NINT8** | `8.2979` | INT8 + gs48 affine + per-neuron scale/min | 高保真端点、敏感权重 |
| **NINT6** | `6.5896` | INT6 + gs24 affine + per-neuron scale/min | 高质量中间档 |
| **NINT5** | `5.5063` | INT5 + gs28 affine + per-neuron scale/min | 质量与体积折中 |
| **NINT4** | `4.5063` | INT4 + gs24 affine + per-neuron scale/min | 常规主力格式 |
| **NINT2** | `2.6313` | INT2 + gs16 affine + per-neuron scale/min | Q2_K 同码率标量整数档 |
| **NVQ3J** | `3.0464` | D4/4D + 2-bank joint state | 约 3 bit 的质量优先档 |
| **NVQ3J-L** | `~3.55` | D4/4D + 10-bit index + joint state | 对位 IQ3_S 的扩展 VQ 档 |
| **NVQ3** | `3.0459` | D4/4D 格点 + parity sign | 固定格点的约 3 bit 基线 |
| **NVQ2J-XL** | `~2.58` | E8/8D + 12-bit index + joint state | 对位 IQ2_S 的扩展 VQ 档 |
| **NVQ2J-L** | `~2.30` | E8/8D + 10-bit index + joint state | 对位 IQ2_XS 的扩展 VQ 档 |
| **NVQ2J** | `~2.047` | E8/8D + joint scale/codebook state | 约 2 bit 的质量优先档 |
| **NVQ2** | `2.0459` | E8/8D 格点 + parity sign | 固定格点的约 2 bit 基线 |
| **NVQ1-L** | `~1.56` | 8D 2048-entry ternary VQ | 1.x bit 质量档 |
| **NVQ1-S** | `~1.34` | gs24 双 bank ternary VQ9 | 1.x bit 紧凑档 |
| **NPQ0-L** | `1.00435` | state-conditioned PQ3+4 | 约 1 bit 档 |
| **NPQ0-S** | `0.83755` | state-conditioned PQ3+3 | 当前最低码率档 |
| **TPQ-X** | `~1.0` | 8D、256-entry 共享学习码本 | TPQ 最紧凑专家档 |
| **TPQ-W** | `~2.0` 物理 / `1.5` 逻辑 | 8D、4096-entry 共享学习码本 | 12-bit 索引质量档 |
| **TPQ-V** | `~2.0` | 4D、256-entry 共享学习码本 | 细粒度专家档 |
| **TPQ-VV** | `~4.0` 物理 / `3.0` 逻辑 | 4D、4096-entry 共享学习码本 | TPQ 高质量专家档 |
| **TPQ-I4G64** | `4.25` | 对称 INT4、FP16 group scale | TPQ dense 权重 |

21 种基础编码均已接入对应的 MFQ save/load/mmap 与生产运行时路径。NINT、NVQ、
NPQ、NEPQ 已接入 CUDA、MLX/Metal 与 C++ native loader；Metal 路径包含
group/vec 解码 GEMV、qmv_wide 小 M MMQ 与在线解码 `simdgroup_matrix` GEMM，
M≥64 默认临时反量化后调用 MLX dense GEMM，不保留第二份模型权重。兼容的
NINT/VQ/NPQ/NEPQ gate/up 可直接融合 SwiGLU。TPQ 的 Metal/C++ 路径沿用既有
CCCP payload 兼容层，并接受新的 TPQ dtype 名称。

## 4 种跨专家派生编码

| 格式 | 基础编码 | 典型 bpw | 状态 |
|---|---|---:|---|
| **NEPQ1-L** | NVQ1-L | `1.63021` | 文件、量化、CUDA、Metal、C++、NINTM |
| **NEPQ1-S** | NVQ1-S | `1.42057` | 文件、量化、CUDA、Metal、C++、NINTM |
| **NEPQ0-L** | NPQ0-L | `1.08647` | 文件、量化、CUDA、Metal、C++、NINTM |
| **NEPQ0-S** | NPQ0-S | `0.91947` | 文件、量化、CUDA、Metal、C++、NINTM |

四种派生编码已接入 native C++ loader 与 NINTM mixed-family grouped runtime，整模 KLD 尚待验证。
`NEPQ0-SR2` 是 coarse-bank + 2-bit residual selector 的实验候选，典型码率 `1.0103 bpw`；它的位流结构
不同于 `NEPQ0-S`，正式接入前不计入上述 17 种格式。

## NINT

NINT 为每个输出神经元保存一组 FP16 顶层 scale/min，并为短 group 保存低位相对 scale/min：

```text
w[j, i] ~= d_neuron[j] * local_scale[j, g] * q[j, i]
           - dmin_neuron[j] * local_min[j, g]
```

| 格式 | Profile `(bits, gs, sub_bits)` | K=5120 bpw |
|---|---:|---:|
| NINT2 | `(2, 16, 5)` | `2.6313` |
| NINT4 | `(4, 24, 6)` | `4.5063` |
| NINT5 | `(5, 28, 7)` | `5.5063` |
| NINT6 | `(6, 24, 7)` | `6.5896` |
| NINT8 | `(8, 48, 7)` | `8.2979` |

per-neuron 顶层参数避免了 K-quant 每 256 个权重重复保存 super-block scale/min。节省出的预算可用于
更短的局部 group。NINT4 在基本相同的 `4.5 bpw` 下使用 gs24，而 Q4_K 使用 32-weight sub-block。

NINT2 通过 `groupsize × scale_bits × min_bits` 三维非支配搜索确定为 `(16, 5, 5)`。
10 个 Qwen3.6-27B 真实矩阵、每矩阵 256 行的验证中，NINT2 为
`2.6291 bpw / 11.1920 dB`，llama.cpp Q2_K 为 `2.6250 bpw / 10.4035 dB`。
CUDA decode 使用 `dp4a`，中等 batch 将两个 gs16 组共享一个 K32 activation tile 并分别
应用 scale/min；大 batch 根据实测交叉点切换为紧凑解码加 cuBLAS。

Metal 保留同一套 packed 数据流，但映射到 Apple GPU 原语：NINT2/3/4/6/8 直接读取连续
bitstream，NINT5 在上传时转为 low4/high1 两平面；FP16 prefill 只把当前 `K x 64`
权重块在线解码到 threadgroup memory，再由八个 SIMD-group 执行
`simdgroup_matrix` 累加，不保留完整 FP16 权重副本。
NINTM routed decode 会把各 cohort 的位流/LUT 拼接为连续 buffer，并以逐 expert 描述符在
一次异构 Metal dispatch 中只解码 top-k 选中的 expert。带 signed-Hadamard rotation 的
NEPQ 会按 rotation key 复用输入变换，再由 expert 描述符在同一个 grouped dispatch 中选择；
兼容的 gate/up projection 也可在一次 grouped dispatch 中同时计算。

## NVQ

NVQ 用一个 FP16 neuron anchor 表示整行幅度，group state 表示局部幅度，短向量 index 表示方向：

```text
w[j, i] ~= anchor[j] * alpha[state[j, g]]
           * codebook[bank[state]][index] * sign
```

- **NVQ2** 使用 E8/8D 格点。8-bit index 与 7-bit parity sign 编码 8 个权重，天然适合两次 DP4A。
- **NVQ3** 使用 D4/4D 格点。每 4 个权重一个 8-bit index，提供更细的局部方向选择。
- **NVQ2J/NVQ3J** 让原有 4-bit group state 同时选择幅度与 shape bank，主位流几乎不增加。
- **NVQ2J-L/NVQ2J-XL** 分别使用 10/12-bit E8 index；UD recipe 中
  `IQ2_XXS/XS/S` 依次映射到 `NVQ2J/NVQ2J-L/NVQ2J-XL`。
- **NVQ3J-L** 使用 10-bit D4 index，UD recipe 中 `IQ3_S` 映射到该档位。
- **NVQ1-L/NVQ1-S** 使用 ternary 向量。`L` 是较高质量布局，`S` 是更紧凑的布局。

NVQ3J 默认使用 2-bank analytic state：state 的低 1 bit 选择码本，高 3 bit 选择八级幅度。完整
Qwen3.5-9B `8192 x 4096` attention tensor 上，它只比 NVQ3 增加 `0.000504 bpw`，weighted NMSE
降低 `10.83%`；专用 M1 kernel 的延迟开销为 `1.22%`。

## NPQ

NPQ 面向约 1 bit 及以下：

- **NPQ0-L** 将每个 8D 向量分成两个 4D 子向量，分别使用 3-bit 与 4-bit index；gs24 共享
  3-bit state，用于选择码本对和相对 scale。
- **NPQ0-S** 把每个 8D 向量分成两个 4D 子向量，分别使用 3-bit index；gs24 共享 2-bit state，
  从四对 `8 x 4` 子码本和四级 scale 中选择。

两者都保留 FP16 neuron anchor，index/state 流跨 tensor 连续打包，不为 K 写入永久 padding。
NPQ0-S 文件只保存 320 B 的 PQ 因子；loader 在初始化时生成 2112 B 笛卡尔积 decode LUT，kernel
因此仍可用一次 64-bit lookup 读取完整 8D 码字。临时 LUT 不计入模型 bpw。

## NEPQ

NEPQ 为一个逻辑 `[experts,out,K]` projection 共享最多 256 份表。每四个连续 gs24 group 组成一个
96-weight super-group，并保存一个 uint8 bank ID；group state、vector index、FP16 neuron anchor
继续使用对应 NPQ/NVQ 基础格式。未激活 expert 的索引、权重和表均不解码。

```text
bank = bank_id[expert,row,group//4]
w_hat = anchor[expert,row] * scale[bank,state] * code[bank,state,index]
```

代表性 `[256,2048,6144]` projection 的码率为 NEPQ0-S `0.91947`、NEPQ0-L `1.08647`、
NEPQ1-S `1.42057`、NEPQ1-L `1.63021 bpw`。NEPQ0-S 使用 compact 320 B/bank 表池执行小 M，
grouped 路径额外使用加载时生成的 2112 B/bank LUT；模型文件仍只保存 compact 表。

## TPQ

TPQ 是早期 CCCP 格式的正式名称。新 MFQ 文件写入 `TPQ-X/W/V/VV` 和
`TPQ-I4G64`；loader 同时接受旧文件中的 `CCCP-X/W/V/VV` 与
`CCCP-I4G64`。两套名称使用完全相同的 `CPQ1`、`CI41` payload、tier ID、
码本、索引和 kernel，不涉及重新量化。

`W`、`VV` 的码本有 4096 项，逻辑索引为 12 bit。当前 TPQ kernel 使用
`uint16` 物理索引，因此文件与显存中的实际索引开销分别为 `2.0`、`4.0 bpw`。
码本按层、投影和 cohort 共享，其开销由具体矩阵规模决定。

旧 `cccp.json / cccp-1` 目录继续作为可导入和推理的历史工件格式。正式 MFQ
文件和 API 使用 TPQ 名称；CLI 的正式入口是 `mfq tpq`，`mfq cccp` 保留为兼容别名。

## NINTM 通用专家容器

NINTM v2（线格式魔数 `NIM2`）可在同一个逻辑 `[experts,out,K]` 张量中混合全部 24 种
可用于 routed expert 的正式编码。
相同 `family + profile + codebook/table artifact + rotation` 的专家聚成一个 cohort；每个 cohort 保存
全局 expert IDs、dtype、可选运行时元数据和原生 packed payload。它不会把低位权重展开为 FP16，也
不会给每一行附加格式标签。

```text
NIM2 tensor header
for each homogeneous cohort:
    expert_ids
    dtype
    runtime metadata
    native NINT / NVQ / NPQ / NEPQ / TPQ payload
```

Python mmap、HF/GGUF 流式转换器、CUDA grouped decode/prefill 和 C++ runtime 使用同一结构。输入激活按
`(transform, groupsize, group_count)` 量化一次并在兼容 cohort 间复用；NEPQ 的 signed-Hadamard
输入按 rotation key 复用。NVQ2/NVQ2J 的 grouped MoE 路径可在一个 warp 内共享 activation 与
sub-scale state；纯 NINT 容器继续使用异构单次 dispatch 快路径。

### DeepSeek-V4-Flash mixed-family 实例

`DeepSeek-V4-Flash-EW88G-NVQ2J-NINT4.mfq` 使用 NINTM v2 在 routed experts 中混合 NVQ2J 与
NINT4。非专家张量沿用 UD IQ1_S recipe，专家精度由 REAP exposure 与实际量化 NMSE 联合分配。

| 项目 | 数值 |
|---|---:|
| 文件大小 | `87,994,061,055 B` |
| 受跟踪权重平均码率 | `2.34048 bpw` |
| Tensor / NINTM Tensor | `1,285 / 86` |
| Expert projection 选择数 | `22,016` |
| NVQ2J / NINT4 | `20,001 / 2,015` |
| gate/up NINT4 占比 / exposure 覆盖 | `17.25% / 54.20%` |
| down NINT4 占比 / exposure 覆盖 | `1.05% / 90.89%` |

该工件已通过文件结构、mmap 和抽样解码校验。对官方参考模型的独立整模 MeanKLD 尚未完成，
因此当前数据只证明 mixed-family 存储与运行路径可用。

### NEPQ0-SR2 实验格式

正式 `NEPQ0-S` 的四个 gs24 group 共用一张 bank：

```text
coarse_bank = bank_id[expert,row,supergroup]
actual_bank[p] = coarse_bank                    # p = 0..3
```

实验 `NEPQ0-SR2` 在同一个 96-weight super-group 内，为四个 gs24 group 分别增加 2-bit
residual selector：

```text
coarse_bank = coarse_id[expert,row,supergroup]      # 8 bit
residual = residual_id[expert,row,supergroup,p]     # 2 bit
actual_bank[p] = dictionary[expert,coarse_bank,p,residual]
```

这里的 residual 修正 bank ID，不保存权重残差。`residual=0` 固定返回 coarse bank，其余三个值
从 expert-local 条件字典中选择替代 bank。每个 expert/projection 的字典包含
`256 * 4 * 3 = 3072 bytes`。

| 项目 | NEPQ0-S | NEPQ0-SR2 |
|---|---:|---:|
| bank 粒度 | 96 weights | 24 weights，由 coarse bank 约束 |
| selector / 96 weights | 8 bit | 8-bit coarse + 4 x 2-bit residual |
| 典型 bpw | `0.91947` | `1.01029` |
| GLM-5.2 layer40 宏平均 SNR | `5.09827 dB` | `5.62028 dB` |

SR2 条件字典使用 imatrix-weighted group SSE 训练；现有结果还包含 H2048 ADMM、anchor refit 和
一轮选中 bank 码本联合更新，因此 `+0.52200 dB` 是完整训练配置的收益。固定 16-bit selector
优于已测试的 Huffman `16.21 bit` 和 rANS 估计 `16.28 bit`。SR2 尚未实现正式位流、CUDA kernel、
native C++ loader及整模 KLD，因此不计入当前16种正式格式。

## 校准关系

格式编码与精度分配是两个层次：

| 机制 | 改变什么 | 是否新增格式 |
|---|---|---|
| Tensor-wise codebook | 为 NVQ/NPQ tensor 拟合专属码本 | 否 |
| Imatrix / neuron gain | 改变量化目标或吸收逐神经元增益 | 否 |
| 梯度精度校准 | 在候选格式与精度配方中为计算组选择精度 | 否 |
| Expert-Wise Precision | 为每个 routed expert 独立选择任一正式精度家族 | 否 |
| NINTM | 在一个 MoE tensor 中保存多个同质 precision cohort | 否，属于容器 |

BF16/F16/F32 小参数和权重，以及模型原生的 block-FP8/MXFP4 权重，
都属于未经 MFQ 量化的保留精度，不计入上述自定义量化格式。MXFP8/MXFP4 在 MFQ 中
将原始 value 字节和 E8M0 scale 封装为一个自包含记录，不做 FP16 降精度。

## 单文件运行附件

MFQ v2 使用 `__mfq_asset__/` 保留名前缀和 `BLOB` dtype 保存非权重数据：

| Record | 内容 |
|---|---|
| `__mfq_asset__/model_config.json` | 模型结构与运行参数 |
| `__mfq_asset__/tokenizer.gguf` | tensor-free GGUF metadata，包含 tokenizer、chat template、special-token ID 与模型 metadata |

`FileHeader.extra.runtime_assets` 记录每个附件的 media type、字节数与 SHA-256。附件沿用
现有 record table，不改变张量 payload 与 offset 语义；旧 C++ loader 会将未知保留名作为
未访问 record 忽略。新 runtime 从内存 GGUF metadata 初始化 llama.cpp vocab，不生成临时文件。

## 分片容器

MFQ v2 支持与 llama.cpp GGUF 相同的编号形式：
`model-00001-of-00004.mfq`。文件名使用 1 基编号，头部的 `split.no`
使用 0 基编号；每片都保存 `split.count`、`split.tensors.count` 和
`split.records.count`。模型配置、tokenizer 等 `BLOB` 附件只保存在第一片。

Python 与 C++ loader 可接受组内任意一片，自动发现并 mmap 全组文件；加载时校验
片号、总片数、架构、版本、全局记录数、全局张量数、重复记录和缺片。张量 payload
不跨片，分片不改变 dtype、量化参数或 payload 字节。

量化时可直接生成分片：

```text
python -m mfq.tools.quantize_hf_to_mfq ... --output model.mfq --split-max-size 4G
python -m mfq.tools.quantize_gguf_to_mfq ... --output model.mfq --split-max-tensors 128
```

已有完整 MFQ 可流式切分，无需整模读入内存：

```text
python -m mfq.tools.split_mfq --input model.mfq --output model.mfq --split-max-size 4G
```

## 代表性结果

| 实验 | 结果 |
|---|---|
| Qwen3.6-27B 全矩阵 NINT4 vs Q4_K | 基本相同 bpw，NINT4 SNR 高 `0.54-0.57 dB` |
| 五个全矩阵 NINT8 vs Q8_K | NINT8 平均高 `0.79 dB`，同时少 `0.827 bpw` |
| Qwen3.5-9B 十个矩阵 NVQ2 vs IQ2_XXS | `9.620 vs 9.132 dB` |
| Qwen3.5-9B 十个矩阵 NVQ3 vs IQ3_XXS | `14.996 vs 13.225 dB` |
| Qwen3.5-9B 完整 NVQ2J 模型 vs UD IQ2_XXS | 文件小 `3.45%`，KLD 相对降低 `15.33%` |
| 完整 attention tensor NVQ3J vs NVQ3 | 增加 `0.000504 bpw`，weighted NMSE 降低 `10.83%` |
| GLM-5.2 layer40 NEPQ0-S vs 四 bank/row | 增加约 `0.070-0.078 bpw`，held-out SNR 宏平均提高 `0.609 dB` |
| DeepSeek-V4-Flash EW mixed-family | `87.994 GB`，平均 `2.34048 bpw`，NINT4 选择集中覆盖高 exposure 专家 |

项目设计、整模结果和 runtime 数据见 [README](README.md)，部署接口见
[C++ runtime README](cpp_runtime/README.md)。
