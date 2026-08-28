<div align="center">

# TyloQuant MFQ

<img src="./docs/figures/tylogi-ai-lab.svg" alt="Tylogi AI Lab" width="520">

**神经元锚定的混合格式量化与高保真 LLM 推理**

**Every Bit. Maximum Fidelity.**

NINT · NVQ/NPQ · NEPQ · TPQ · Expert-Wise MoE · CUDA · Metal · C++ Runtime

</div>

<p align="center">
  <a href="./README.md">English</a> | <strong>中文</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/Tylogi">Hugging Face 模型主页</a> · <a href="https://www.modelscope.cn/profile/Tylogi">ModelScope 模型主页</a>
</p>

## 核心结果

### DeepSeek-V4-Flash-0731

<img src="./docs/figures/deepseek-v4-flash-mfq-vs-ud-kld.svg" alt="DeepSeek-V4-Flash-0731 MFQ 与 Unsloth Dynamic 模型大小和 Mean KLD 对比" width="100%">

**评测：** 官方 0731 权重，WikiText-2 共 573 个 chunk、146,115 个计分 token，`ctx=512`。

| 发布档位 | 大小 | Mean KLD ↓ | Same-top ↑ |
|---|---:|---:|---:|
| S | 77.519 GiB | `0.313576` | `82.2913%` |
| M | 88.007 GiB | `0.244488` | `84.5300%` |
| L | 98.007 GiB | `0.201444` | `86.0753%` |

**同体积对比：** 与三组大小最接近的 Unsloth Dynamic（UD）基线相比，MFQ 将 Mean KLD 降低 **34.24–51.42%**。

### Qwen3.5-9B：文件大小与 Mean KLD

<img src="./docs/figures/qwen35-9b-mfq-vs-ud-size-kld.svg" alt="Qwen3.5-9B MFQ 与 Unsloth Dynamic 文件大小和原始 Mean KLD 对比" width="100%">

- **评测：** 完整 WikiText-2，共 145 个 chunk、148,335 个计分 token；所有档位使用同一 BF16 参考模型。
- **结果：** 上图每个匹配精度档位中，MFQ 的原始 Mean KLD 均更低。

### Qwen3.6-27B：文件大小与质量

<img src="./docs/figures/qwen36-27b-mfq-vs-ud-size-quality.png" alt="Qwen3.6-27B MFQ 与 Unsloth Dynamic 文件大小、原始 Mean KLD 和 same-top accuracy 对比" width="100%">

- **评测：** 完整 145-chunk 评测，统一使用 `ubatch=2048`；图中每个档位均覆盖全部 chunk。
- **对比：** 每个 MFQ 文件均与对应的同体积 Unsloth Dynamic recipe 配对。
- **指标：** Mean KLD 越低越好，same-top accuracy 越高越好。

## 项目简介

**TyloQuant MFQ**（简称 **MFQ**）联合设计量化格式、精度分配与推理 kernel，面向高保真 LLM 部署。项目支持 `0.84-8.30 bpw` 的自定义权重编码，可按计算组或 MoE 专家分配精度，并通过由 C++ runtime 支撑的原生 CUDA 与 Metal 路径直接执行 packed 权重。

公开 MFQ 模型使用统一的 `V`/`S` 命名：与 llama.cpp `IQ*` 对齐的向量量化模型使用 `V`，与 `Q*_K*` 对齐的标量量化模型使用 `S`。例如：`IQ3_XXS → V3-XXS`，`Q4_K_XL → S4-L`。

## 安装

### 环境要求

#### 通用

- Git
- [uv](https://docs.astral.sh/uv/)
- CMake `>=3.26` 与原生 C++ 工具链

#### 推理后端（二选一）

- **CUDA：** Linux 或 Windows、NVIDIA GPU，以及包含 `nvcc`、cuBLAS 和 CUDA runtime 的 CUDA Toolkit `>=12`。
- **Metal：** Apple silicon 与 macOS；`metal` extra 会提供 MLX 及其原生 runtime 资产。

#### 可选

- 仅在 MFQ 需要从源码构建浏览器 Web UI 时才需要 Node.js 与 npm；缺少它们时，`mfq serve` 仍可提供 API。

`uv` 会在需要时自动配置 MFQ 要求的 Python `>=3.10` runtime，无需单独安装 Python。

### 从源码安装

命令行统一使用 `mfq`。在 PowerShell 中，请将多行命令写成一行，或用反引号替换末尾的 `\`。

```shell
git clone https://github.com/Tylogi/TyloQuant.git MFQ
cd MFQ
```

#### CUDA（Windows 或 Linux）

原生 CUDA 推理不需要 PyTorch 或 LibTorch。

```shell
uv sync --extra daemon
uv run mfq build --backend cuda
```

#### Metal（Apple silicon）

```shell
uv sync --extra daemon --extra metal
uv run mfq build --backend metal
```

需要离线量化或校准时，加入对应 extra：

```shell
# CUDA
uv sync --extra daemon --extra train --extra calibration

# Apple silicon
uv sync --extra daemon --extra metal --extra train --extra calibration
```

`mfq build` 会自动探测操作系统和推理加速器。自定义 CMake 配置参数放在 `--` 之后，例如：

```shell
uv run mfq build -- -DCMAKE_CUDA_ARCHITECTURES=90
```

`mfq build` 会把可执行文件和 CMake 参数写入 `build/mfq-runtime.json`。`mfq serve` 会复用该构建；即使使用了自定义 `--build-dir`，可执行文件丢失时也会按原配置重建。

验证安装：

```shell
uv run mfq --help
uv run mfq build --help
```

## 快速开始

空载启动服务：

```shell
uv run mfq serve
```

打开 <http://127.0.0.1:8090/>，并在另一个终端检查 API：

```shell
curl http://127.0.0.1:8090/health
```

从 [Hugging Face](https://huggingface.co/Tylogi) 或 [ModelScope](https://www.modelscope.cn/profile/Tylogi) 下载 MFQ 文件。可以在 Studio 模型目录中加载，也可以重启服务并传入模型路径：

```shell
uv run mfq serve --model /absolute/path/to/model.mfq
```

模型状态变为 `ready` 后即可开始对话。模型目录、身份认证和 API 用法见 [`mfq serve`](./docs/cli/serve.md)。

## 量化

`mfq quantize` 接受 HF safetensors 目录、满精度 MFQ 或满精度 GGUF。仅执行量化时，安装 `train` extra：

```shell
uv sync --extra train
```

将 HF checkpoint 量化为统一 NINT4：

```shell
uv run mfq quantize model-hf model-NINT4.mfq \
  --bits 4 --groupsize 24 --sub-bits 6 --backend auto
```

不做量化，直接把原始 checkpoint 写入满精度 MFQ：

```shell
uv run mfq quantize model-hf model-full.mfq --full-precision
```

混合 GGUF recipe、Expert-Wise override、MTP 补全、Important Neurons、分片和断点续作见 [`mfq quantize`](./docs/cli/quantize.md)；activation imatrix 和校准见 [`mfq calibrate`](./docs/cli/calibrate.md)。

## Web UI

`mfq serve` 启动公开 API 和 Web UI，并管理私有 C++ worker。可以空载启动，也可以直接加载模型：

```shell
uv run mfq serve
uv run mfq serve --model path/to/model.mfq --host 127.0.0.1 --port 8090
uv run mfq serve --model-dir path/to/models --host 127.0.0.1 --port 8090
```

`--host` 和 `--port` 控制公开 API 监听地址，默认是 `127.0.0.1:8090`。打开命令输出的 Web UI 地址即可。

桌面版 Studio 的 **Models and jobs** 页面可直接加载本地 `.mfq` 文件，不会复制模型。选择任一分片时会加载完整的 sibling shard family。

<img src="./docs/figures/tyloquant-mfq-webui-zh.png" alt="TyloQuant MFQ 本地推理 WebUI" width="100%">

## 工作原理

| 层次 | 设计 | 作用 |
|---|---|---|
| 权重格式 | NINT、NVQ、NPQ、NEPQ、TPQ | 提供从 8 bit 到低于 1 bit 的质量档 |
| Dense 分配 | 重要性感知分配 | 按计算组选择精度 |
| MoE 分配 | Expert-Wise Precision | 将码率集中到期望输出贡献更高的专家 |
| 专家容器 | NINTM v2 | 在同一 MoE tensor 中保存异构格式 |
| Runtime 资产 | Reserved `BLOB` records | 在 MFQ 文件中保存 config、tokenizer、chat template 与 special-token metadata |
| 推理 | CUDA/Metal kernels + C++ runtime | 不展开完整 FP16 权重，直接执行 packed 权重 |

NINT 在每个输出行上共享仿射元数据，把空间留给更短的局部 group。低于 4 bit 时，NVQ、NPQ 与 NEPQ 使用短向量编码和专家感知的码本共享。分配器按序列化后的实际字节数选择格式，NINTM 将兼容专家分组后直接执行 packed 权重。

## 当前支持范围

> [!NOTE]
> Runtime 仍处于实验阶段。CUDA 默认使用单 GPU。

- `端到端`：原生 MFQ 加载、prefill、decode 与 generation。
- `部分支持`：已有架构相关组件，但没有完整的公开模型 Runtime。
- `专用路径`：使用独立的 TPQ/MFQ 执行路径。

| 模型或系列 | 转换与封装 | CUDA 推理 | Metal 推理 | 支持范围或当前限制 |
|---|---|---|---|---|
| Qwen3.5 | HF/GGUF/满精度 MFQ | 端到端 | 端到端 | Full/linear hybrid CausalLM |
| Qwen3.6 | HF/GGUF/满精度 MFQ | 部分支持 | 部分支持 | Routed-MoE 组件；暂无公开端到端 Runtime |
| Gemma4 | HF/GGUF 与 MFQ 分片 | 端到端 | 端到端 | Mixed full/sliding attention |
| DeepSeek-V4-Flash | HF/GGUF、Expert-Wise MFQ 与 TPQ | 端到端 | 端到端 | Compression、indexer 与 sparse-attention 路径 |
| MiniCPM-o 4.5 | 官方 composite HF graph 转 MFQ | 端到端 | 端到端 | 运行时需要官方模型目录 |
| GLM-MoE-DSA | 原生 MFQ 与混合精度 | 端到端 | 端到端 | Dense/sparse MLA 路径 |
| Kimi-K3 | TPQ/MFQ 封装 | 专用路径 | 专用路径 | 仅使用 TPQ/MFQ 执行路径 |

通用能力：

- NINTM v2 流式转换；
- 自包含与分片 MFQ 文件；
- Python/C++ 加载；
- 支持 SSE 的 OpenAI 兼容 API。

Kernel 和模型细节：[Runtime 支持矩阵](./docs/runtime-support.md)。

## 文档

### CLI 与服务

- [`mfq build`](./docs/cli/build.md)
- [`mfq quantize`](./docs/cli/quantize.md)
- [`mfq calibrate`](./docs/cli/calibrate.md)
- [`mfq serve` 与模型管理](./docs/cli/serve.md)
- [HTTP API](./docs/api/http.md)
- [WebSocket API](./docs/api/websocket.md)

### Runtime 与验证

- [Runtime 支持矩阵](./docs/runtime-support.md)
- [Runtime sampling profile](./docs/runtime-sampling-profiles.md)
- [原生 CUDA Runtime 验证](./docs/cuda-native-runtime-validation.md)

### 量化与模型集成

- [Expert-Wise 联合预算求解器](./docs/ew-joint-solver.md)
- [MiniCPM-o 4.5 Runtime](./docs/minicpmo45.md)
- [自包含 release](./docs/release.md)

## 致谢

MFQ 使用或参考了以下项目：

- [llama.cpp](https://github.com/ggml-org/llama.cpp)：GGUF 互操作、推理后端和评测工具。
- [oMLX](https://github.com/jundot/omlx)：Apple silicon 性能与 Runtime 设计参考。
- [MLX](https://github.com/ml-explore/mlx)：macOS 路径使用的 array framework 与 Metal runtime。
- [PyTorch](https://github.com/pytorch/pytorch) 与 [Transformers](https://github.com/huggingface/transformers)：模型接入与量化基础设施。
- [Unsloth](https://github.com/unslothai/unsloth)：用于对比的 Dynamic 量化模型。

许可证：[Apache License 2.0](./LICENSE)。
