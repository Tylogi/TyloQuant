<div align="center">

<p align="center">
  <a href="./README.md">English</a> | <strong>中文</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/Tylogi">Hugging Face 模型主页</a> · <a href="https://www.modelscope.cn/profile/Tylogi">ModelScope 模型主页</a>
</p>

## 核心结果

### DeepSeek-V4-Flash-0731

<img src="./docs/figures/deepseek-v4-flash-mfq-vs-ud-kld.svg" alt="DeepSeek-V4-Flash-0731 MFQ 与 Unsloth Dynamic 模型大小和 Mean KLD 对比" width="100%">

官方 0731 权重的 WikiText-2 评测使用 `ctx=512`，覆盖 573 个 chunk 和 146,115 个计分 token。发布版 77.519 GiB S 档的 Mean KLD 为 `0.313576`，same-top 为 `82.2913%`；88.007 GiB M 档为 `0.244488` / `84.5300%`；98.007 GiB L 档为 `0.201444` / `86.0753%`。三组近似同体积 UD 对位中，MFQ 将 Mean KLD 降低 **34.24–51.42%**。

### Qwen3.5-9B：文件大小与 Mean KLD

<img src="./docs/figures/qwen35-9b-mfq-vs-ud-size-kld.svg" alt="Qwen3.5-9B MFQ 与 Unsloth Dynamic 文件大小和原始 Mean KLD 对比" width="100%">

完整 WikiText-2 评测包含 145 个 chunk、148,335 个计分 token，并使用同一 BF16 参考模型。上图所示的每个匹配精度档位中，MFQ 的原始 Mean KLD 均更低。

### Qwen3.6-27B：文件大小与质量

<img src="./docs/figures/qwen36-27b-mfq-vs-ud-size-quality.png" alt="Qwen3.6-27B MFQ 与 Unsloth Dynamic 文件大小、原始 Mean KLD 和 same-top accuracy 对比" width="100%">

对齐的 `ubatch=2048` 评测比较当前同体积 MFQ 文件与对应 Unsloth Dynamic recipe。Mean KLD 越低越好，same-top 越高越好。图中所有档位均使用当前文件的完整 145-chunk 评测。

## 项目简介

**TyloQuant MFQ**（简称 **MFQ**）联合设计量化格式、精度分配与推理 kernel，面向高保真 LLM 部署。项目支持 `0.84-8.30 bpw` 的自定义权重编码，可按计算组或 MoE 专家分配精度，并由 CUDA kernel 与 C++ runtime 直接执行 packed 权重。

公开 MFQ 模型使用统一的 `V`/`S` 命名：与 llama.cpp `IQ*` 对齐的向量量化模型使用 `V`，与 `Q*_K*` 对齐的标量量化模型使用 `S`。例如：`IQ3_XXS → V3-XXS`，`Q4_K_XL → S4-L`。

## 安装

### 环境要求

- Python `>=3.10`
- Git
- [uv](https://docs.astral.sh/uv/)
- CMake 与原生 C++ 工具链
- 浏览器 Web UI 需要 Node.js 与 npm
- Windows 或 Linux 上的 CUDA 加速需要 NVIDIA GPU 与 CUDA Toolkit
- macOS 上的 Metal 加速需要 Apple silicon

### 从源码安装

`mfq` 是唯一的公开命令行入口。Windows PowerShell、macOS Terminal 和 Linux shell 使用相同的命令：

```shell
git clone https://github.com/mfq/mfq.git MFQ
cd MFQ
# CUDA 推理（Windows 或 Linux；不需要 PyTorch/LibTorch）
uv sync --extra daemon

# Metal 推理（Apple silicon）
uv sync --extra daemon --extra metal

# 编译得到可执行文件
uv run mfq build
```

训练和校准是独立开发流程。只在需要执行这些流程的机器上安装对应 extra：

```shell
uv sync --extra train --extra calibration
```

`mfq build` 会自动探测操作系统和推理加速器。自定义 CMake 配置参数放在 `--` 之后，例如：

```shell
uv run mfq build -- -DCMAKE_CUDA_ARCHITECTURES=90
```

成功构建后，`mfq` 会把实际产物及其构建配方记录在托管清单中。即使使用 `--build-dir` 指定了自定义目录，后续 `mfq serve` 也会自动找到该产物；产物丢失时会按相同配置重建。

验证安装：

```shell
uv run mfq --help
uv run mfq quantize --help
```

### 量化

`mfq quantize` 是稳定的统一量化入口。它会自动识别 HF safetensors 目录、满精度 MFQ 或满精度 GGUF，并直接调用现有生产量化器。

`mfq calibrate` 会生成校准产物，包括可通过 `mfq quantize --imatrix` 复用的重要性矩阵。

```shell
# 把 HF 原生存储精确写入满精度 MFQ：BF16 仍是 BF16；
# block FP8/MXFP4 及其 E8M0 scale 自包含并保持精确。
# 该命令不执行 MFQ 量化。
uv run mfq quantize model-hf model-full.mfq --full-precision

# 满精度 MFQ 再复用同一套 NINT/VQ/NPQ/NEPQ/TPQ 量化路径。
uv run mfq quantize model-full.mfq model-NINT3.mfq \
  --bits 3 --groupsize 24 --sub-bits 5 --backend cpu --device cpu

# BF16 GGUF 来源，按混合 recipe 量化
uv run mfq quantize model-bf16.gguf model-S4-L.mfq \
  --recipe UD-Q4_K_XL.gguf --imatrix imatrix.gguf \
  --q8-mode nint8-0 --device cuda

# HF 来源，叠加逐专家精度方案（EW）
uv run mfq quantize model-hf model-EW.mfq \
  --recipe UD-Q4_K_XL.gguf --ew-scheme expert-precision.json

# 从原始 BF16 checkpoint 给既有量化模型补齐完整 MTP head。
# 主干 blob 逐字节保持不变；每个 MTP decoder projection
# 跟随对应主干最后一层的精度。
uv run mfq quantize model-hf model-with-MTP.mfq \
  --base-mfq model-quantized.mfq --backend metal --device mps

# 从准备好的语料收集可复用 activation imatrix。CUDA 默认使用 FP64 累加；
# Apple silicon 使用 BF16 forward + FP32 Metal 累加。输出可直接传给
# `mfq quantize --imatrix`。
uv run mfq calibrate imatrix \
  --model model-hf --corpus calibration-corpus \
  --output calibration.imatrix --backend cuda

uv run mfq calibrate imatrix \
  --model model-hf --corpus calibration-corpus \
  --output calibration.imatrix --backend metal

# Important Neurons（IN）；可从 recipe 自动读取层数
uv run mfq quantize model-bf16.gguf model-IN.mfq \
  --recipe UD-Q4_K_XL.gguf --imatrix imatrix.gguf \
  --important-neurons 1024 --target-size 15G
```

## Web UI

`mfq serve` 会按需构建或更新 runtime，然后启动 API 和 Web UI。可以空载启动并从 catalog 加载模型，也可以在命令行传入初始模型：

```shell
uv run mfq serve
uv run mfq serve --model path/to/model.mfq --host 127.0.0.1 --port 8090
uv run mfq serve --model-dir path/to/models --host 127.0.0.1 --port 8090
```

`--host` 和 `--port` 控制公开 API 监听地址，默认是 `127.0.0.1:8090`。打开命令输出的 Web UI 地址即可。

桌面版 Studio 的 Models and jobs 页面还可以选择并加载任意本地 `.mfq` 文件。Studio 只在私有模型目录中登记所选路径，不会复制模型；选择任一分片时会加载同目录下完整的 sibling shard family。

<img src="./docs/figures/tyloquant-mfq-webui.jpg" alt="TyloQuant MFQ 本地推理 WebUI" width="100%">

## 工作原理

| 层次         | 设计                       | 作用                                                                         |
| ------------ | -------------------------- | ---------------------------------------------------------------------------- |
| 权重格式     | NINT、NVQ、NPQ、NEPQ、TPQ  | 提供从 8 bit 到低于 1 bit 的质量档                                           |
| Dense 分配   | 重要性感知分配             | 按计算组选择精度                                                             |
| MoE 分配     | Expert-Wise Precision      | 将码率集中到期望输出贡献更高的专家                                           |
| 专家容器     | NINTM v2                   | 在同一 MoE tensor 中保存异构格式                                             |
| Runtime 资产 | Reserved`BLOB` records   | 在 MFQ 文件中保存 config、tokenizer、chat template 与 special-token metadata |
| 推理         | CUDA kernels + C++ runtime | 不展开完整 FP16 权重，直接执行 packed 权重                                   |

NINT 沿输出神经元权重行共享顶层仿射元数据，并把节省的预算用于更短的局部 group。低于 4 bit 时，NVQ、NPQ 与 NEPQ 使用短向量编码和专家感知的码本共享。分配器依据真实字节预算选择格式，NINTM 再将兼容专家分组并直接执行 packed 权重。

## 当前支持范围

- Qwen3.5 full attention 与 linear attention/GDN
- Qwen3.6 routed MoE
- Gemma4 GeGLU、sliding-window attention 与 routed MoE
- DeepSeek-V4-Flash HCA/CSA/mHC 与 MoE
- MiniCPM-o 4.5 official composite Python graph with MFQ-backed CUDA matrix modules
- NINTM v2 mixed-family HF/GGUF 流式转换
- 自包含 MFQ 文件，内嵌 runtime config 与 GGUF tokenizer metadata
- 编号 MFQ 分片，支持量化器直接输出与 Python/C++ 透明加载
- 支持 SSE 的 OpenAI 兼容 chat/completions API

当前 CUDA 推理实现仍是单 GPU 研究原型。Apple silicon 现已提供 packed NINT/NVQ/NPQ/NEPQ group-vectorized GEMV、qmv_wide 小 M MMQ、在线解码 `simdgroup_matrix` GEMM，以及在测得的大 M 交叉点采用 CUDA 风格的临时反量化 + MLX GEMM。兼容 NINT/VQ-family gate/up projection 可融合 SwiGLU。MHA/GQA/MQA、动态和滑窗 KV cache、单次 dispatch 的异构 NINTM/NEPQ routing、融合 SSM/GDN kernel、GLM DSA/sparse MLA、DeepSeek-V4 compression/indexer/sparse-attention/HC kernel，以及 Qwen3.5 full/linear hybrid CausalLM prefill/decode/generation runtime 已可用。普通混合格式 QKV 和 FFN projection group 可在一次异构 Metal dispatch 中执行。TPQ-I4G64、TPQ-X/W/V/VV、带 p8-p16 index 的三投影 TPQ-P kernel，以及 Kimi-K3 KDA/MLA、Attention-Residual、SiTU MoE、cache 和 generation graph 也已接入。DeepSeek-V4 已支持原生 MFQ 加载、compressed/local/indexer cache、mmap-backed bounded expert residency、prefill、decode 和 generation。Gemma4 已支持自包含 sharded-MFQ 加载、mixed full/sliding attention、fused norm/GeGLU/MoE、cache 和 generation。GLM-MoE-DSA 已支持原生加载、shared indexer state、dense/sparse MLA、cache 和 generation。Greedy、softmax、top-k/top-p 与 sampling-penalty kernel 仍驻留在 GPU 上。

## 文档

- [MiniCPM-o 4.5 Python Runtime](./docs/minicpmo45.md)
- [Expert-Wise 联合预算求解器](./docs/ew-joint-solver.md)
- [`mfq serve` 与模型管理](./docs/cli/serve.md)
- [自包含 release](./docs/release.md)

## 致谢

MFQ 得益于开源 AI 社区的优秀工作，特别感谢：

- [llama.cpp](https://github.com/ggml-org/llama.cpp)：提供 GGUF 生态、高性能推理后端与评测工具，为 MFQ 的互操作和验证奠定了重要基础。
- [oMLX](https://github.com/jundot/omlx)：其出色的 Apple silicon 推理工程为 MFQ 提供了宝贵的性能基线与设计参考。
- [MLX](https://github.com/ml-explore/mlx)：提供 MFQ 原生 macOS 路径使用的 Apple silicon array framework 与 Metal runtime。
- [PyTorch](https://github.com/pytorch/pytorch) 与 [Transformers](https://github.com/huggingface/transformers)：提供核心研究、模型接入与量化基础设施。
- [Unsloth](https://github.com/unslothai/unsloth)：公开 Dynamic 量化模型与可复现的对比基线。

感谢所有维护者与贡献者，让高性能本地推理更开放、更易用。

许可证：[Apache License 2.0](./LICENSE)。
