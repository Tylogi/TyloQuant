<div align="center">

# TyloQuant MFQ

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./docs/figures/tylogi-ai-lab-lockup-dark.svg">
  <img src="./docs/figures/tylogi-ai-lab-lockup-light.svg" alt="Tylogi AI Lab" width="420">
</picture>

### 新一代量化与推理基础设施

**每一比特，极致保真。**

MFQ 将面向实用率失真前沿设计的神经网络感知 SQ/VQ 格式、高质量的
细粒度混合精度校准量化，以及高效的 CUDA/Metal 推理融为一体。我们的
目标是在内存、存储和延迟约束内，让现有硬件运行尽可能高质量的模型。

<p>
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="Apache 2.0 许可证">
  <img src="https://img.shields.io/badge/runtime-C%2B%2B-black" alt="C++ 运行时">
  <img src="https://img.shields.io/badge/backends-CUDA%20%7C%20Metal-6b57ff" alt="CUDA 和 Metal">
</p>

<p>
  <a href="./README.md">English</a> · <strong>中文</strong>
</p>

<p>
  <a href="#deepseek-v41-raw-hf">DeepSeek V4.1</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#功能特性">功能特性</a> ·
  <a href="#模型">模型</a> ·
  <a href="#性能与质量">性能与质量</a> ·
  <a href="./docs/README.md">文档</a>
</p>

</div>

<a id="deepseek-v41-raw-hf"></a>

## 单台 Mac Studio 直跑 DeepSeek V4.1 raw-HF

> **在一台配备 512 GB 统一内存的 Apple M3 Ultra 上，直接加载并运行
> 476 GB 的 DeepSeek V4.1 Flash Hugging Face Safetensors，无需先转换为
> MFQ 容器。** 针对 Metal 优化的路径将主干网络与 MoE 专家完整常驻统一
> 内存，仅把 Engram 存储卸载到 Mac 内置 SSD。

| 模型 / 硬件 | 放置方式 | 模型加载 | 预填充（约 0.5K / 2K / 16K tokens） | TG，关闭 MTP | TG，高接受率 MTP |
| --- | --- | ---: | ---: | ---: | ---: |
| DeepSeek V4.1 Flash raw-HF / Mac Studio M3 Ultra，512 GB | 全量常驻；仅 Engram 使用内置 SSD | **38.6 秒** | **390.2 / 458.6 / 471.1 tok/s** | **18.29 tok/s** | **33.46 tok/s**（**+83.0%**） |

*本机实测口径：batch size 1、32,768-token context、temperature 0、热态
执行。对于 M3 Ultra 上这一全量常驻的精确模型几何，MFQ 现会自动选择最大
5,440-token prefill chunk；5,440 × 6 条专家路由恰好保持在 MLX 的 32,768-row
sorted-MXFP4 边界以内；同时融合 window-KV 的 RMSNorm、partial RoPE 与
activation fake quantization。组件级 A/B 中，`q_kv_prepare` 耗时降低 1.18%，
整体 evaluated prefill time 降低 0.15%。Prefill 数据是 509、2,009 和
15,969-token 确定性重复型 raw-completion 提示的代表性热态实测值。模型加载
后进程 RSS 约为 287 GiB；440 GiB wired budget 是上限，并非稳态占用。MTP
数据来自确定性的数字续写场景，草稿 token 接受数为 132/132，且输出与关闭
MTP 时完全一致；MTP 收益取决于具体负载，macOS 内存压力也可能造成一定波动。*

当推测不再有效时，MTP 会自适应退出：在一个接受率为 50% 的代码场景中，
开启 MTP 为 **17.52 tok/s**，关闭 MTP 为 **17.59 tok/s**；连续两个低接受率
周期后会退出 MTP，同时保持确定性输出。

### DeepSeek V4.1 TODO

- [ ] **扩大 MTP 的生产场景收益。** MTP 现已支持确定性验证、状态快照、
  自适应深度和低接受率回退。后续继续降低验证开销，并提升自然语言与代码
  场景的接受率，使高接受率场景中的加速覆盖代表性负载，再考虑默认启用。
- [ ] **继续优化 M3 Ultra。** 提升长提示 prefill 与逐 token 生成速度，降低
  模型加载和内存压力开销，并进一步用计算覆盖 Engram cache miss 的内置
  SSD I/O。

MFQ 覆盖从源模型到可部署打包模型的完整流程。它测量激活与损失敏感度，
在精确的序列化大小预算内，按张量、专家和投影粒度分配精度，将结果保存为
自包含的 `.mfq` 容器，并通过优化的 C++ 内核直接执行打包权重。量化格式与
运行时协同设计，使每一比特带来的保真度收益能够真正转化为高速 CUDA/Metal
推理和一条命令即可启动的服务。

<p align="center">
  <img src="./docs/figures/tyloquant-mfq-webui-zh.png" alt="MFQ Studio 运行本地模型" width="900">
</p>

## 快速开始

### 使用预编译包

Apple 芯片上最简单的使用方式是安装预编译的 **MFQ Studio**。预编译包会在
可用时发布到项目的 [Releases](https://github.com/Tylogi/TyloQuant/releases)，
其中包含桌面控制台、MFQ Server 和 C++ 运行时。打开 Studio，注册模型目录，
即可加载任意受支持的 `.mfq` 模型。

也可以直接为服务器指定预编译的 C++ worker：

```bash
uv run mfq serve \
  --running-executable /path/to/mfq-decode-metal \
  --model /models/model.mfq
```

### 从源码构建

MFQ 需要 Git、[uv](https://docs.astral.sh/uv/)、CMake 3.26+ 和 C++ 工具链。
NVIDIA 平台使用 CUDA 12+，Apple 芯片使用 Metal。

```bash
git clone https://github.com/Tylogi/TyloQuant.git MFQ
cd MFQ

# NVIDIA / CUDA
uv sync --extra daemon
uv run mfq build --backend cuda

# Apple 芯片 / Metal
uv sync --extra daemon --extra metal
uv run mfq build --backend metal
```

启动一个暂未加载模型的本地服务器，并打开 <http://127.0.0.1:8090/>：

```bash
uv run mfq serve
```

更详细的环境要求和自定义 CMake 选项请参阅
[`mfq build` 指南](./docs/cli/build.md)。

## 量化并运行模型

使用 MFQ 的 **`S4-M` 混合精度预设**。它会为敏感张量保留更高精度，并默认
保持视觉和 MTP 组件的源精度。

```bash
# 量化依赖
uv sync --extra train

# Hugging Face safetensors -> 自包含 MFQ
uv run mfq quantize \
  /models/Qwen3.8-27B \
  /models/Qwen3.8-27B-MFQ-S4-M.mfq \
  --preset S4-M \
  --backend auto
```

加载并启动量化后的模型：

```bash
uv sync --extra daemon                 # Apple 芯片还需添加 --extra metal
uv run mfq build --backend auto
uv run mfq serve \
  --model /models/Qwen3.8-27B-MFQ-S4-M.mfq \
  --context-size 32768
```

MFQ 还可以量化满精度 MFQ 和 GGUF 源模型，读取逐张量或逐专家精度映射及激活
重要性矩阵，并输出分片模型。详见
[`mfq quantize`](./docs/cli/quantize.md) 和
[`mfq calibrate`](./docs/cli/calibrate.md)。

## 功能特性

### 量化

- **神经元锚定的 SQ 和 VQ 基础格式。** NINT、NVQ、NPQ 和 NEPQ 的每个短
  子分组仍保留各自的缩放与偏置。MFQ 对通用二级缩放机制的改进，是将其中的
  高精度缩放与偏置参数按神经元仅保存一组，从而以更少的元数据 bit 表达同一
  层次。节省出的 bit 可在相同 BPW 下用于更小的 group size，或者更高的子分组
  精度，使基础格式本身获得更高的 SNR 与更低的 SSE，而不依赖内部精度分配
  方法或跨张量位宽分配策略。在真实模型权重上，它们也在几乎所有测试中优于
  其他主流量化格式。

- **最高质量的免校准、免 QAT 量化。** MFQ 持续吸收 DSQ 等现代高效方法，
  在无需校准数据或训练的一键量化场景中，将基础量化质量推到尽可能高的水平。

- **高质量校准模型。** MFQ 提供高质量的细粒度混合精度校准量化，在每个目标
  体积下尽可能保留源模型质量。发布的校准模型会在一致的模型级协议下与源模型
  对比验证。

- **亚专家级和亚张量级混合精度。** MFQ 可以为每个专家中的每个张量分配
  不同精度，将单个 FFN 切分为多个不同精度的 Chunk，或者以更高精度保留其中
  最重要的 N 个神经元。
  每种选择都按实际序列化大小精确计入预算，并由专用的批处理与融合内核高效
  执行这些异构布局。这些能力已经贯通生产推理，而不只停留在理论分配方案或
  研究原型中。

- **量化格式反向传播与训练器。** MFQ 为量化格式提供反向传播算子和开箱即用
  的训练工具，方便进行 QAT 与量化模型微调。

### 推理

- **高效打包权重内核。** 后端专用的 GEMV、小 M MMQ 和大 M 路径直接使用
  MFQ 打包存储，无需常驻 FP16 副本，从而降低预填充、解码和 MTP 工作负载中
  的内存流量。

- **高效异构 MoE 执行。** MFQ 将每个专家投影的逻辑精度与物理内核实现解耦。
  运行时会针对模型中实际存在的精度组合编译并分派融合路径，使异构
  Gate/Up/Down 执行仍保持高效。

- **优化的 CUDA 与 Metal 后端。** 共享的 C++ 模型图与运行时契约驱动后端
  专用内核，在 NVIDIA 和 Apple 硬件上高效执行量化线性代数、注意力、卷积与
  循环状态以及缓存操作。

- **广泛且与时俱进的架构支持。** MFQ 覆盖多种 Dense、MoE、多模态、循环与
  稀疏注意力模型系列，并紧跟重要模型发布，第一时间支持重要的新模型架构。

- **连续批处理。** 请求到达和完成时动态接入请求并压缩活跃序列，结合分页 KV
  存储和可复用执行图，持续提高并发生成效率。

- **RAM/SSD 前缀 KV Cache。** 精确的前缀块可以保留在高速内存层或持久化 SSD
  层，使后续请求乃至进程重启后都能复用已经完成的预填充计算。

- **可组合的多模态与 MTP 执行。** 视觉、音频、实时全双工和草稿组件由模型图
  自动启用并共享同一套运行时，无需为每种架构维护独立的服务栈。

- **超越内存容量的 SSD 流式推理。** 混合 LRU/LFU Cache 让近期使用和高频路由
  的专家常驻内存，路由感知预取则通过并行 I/O 合并冷数据读取。双缓冲暂存和
  依赖感知调度用有效计算覆盖 SSD 延迟；带宽自适应策略会根据命中率与存储
  吞吐，在常驻、预取和按需专家之间分配工作。这样既能限制 RAM 或 VRAM 占用，
  又不会让每个路由层都退化为同步磁盘等待。

| 模型 | 模型大小 | 专家预算 | 精度 | 硬件 | 预填充 | 解码 |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| DeepSeek-V4-Flash-0731 | ~160 GiB | 85.3 GiB | 官方原生 QAT 精度 | Apple M5 Max, 128 GB | **312.4 tok/s** | **18.6 tok/s** |

MFQ 在当前支持的两种精度粒度下都能保持高效的异构 MoE 执行：

| 模型 | 精度粒度 | 平均 BPW | 预填充 | 解码 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.8-Flash-Next | 每个专家一种精度 | **5.80** | **1.13K tok/s** | **25.4 tok/s** |
| Qwen3.8-Flash-Next | 每个专家的 Gate/Up/Down 分别使用独立精度 | **5.79** | **1.12K tok/s** | **24.4 tok/s** |

*Apple M5 Max（40 核 GPU、128 GB）上的预热端到端吞吐。*

### 未来方向

MFQ 是一个开放且持续演进的项目。我们会不断探索、验证并吸收新的量化与推理
技术，通过严格校准带来质量更高的量化模型和更好的本地推理体验。我们也非常
欢迎社区参与贡献，详见[贡献指南](./CONTRIBUTING.md)。

## 模型

已发布的 `.mfq` 模型可从
[Hugging Face](https://huggingface.co/Tylogi) 和
[ModelScope](https://www.modelscope.cn/profile/Tylogi) 下载。

| 模型 | 已发布精度系列 | 下载 |
| --- | --- | --- |
| DeepSeek-V4-Flash-0731 | 逐专家 `V2` 档位 | [Hugging Face](https://huggingface.co/Tylogi/DeepSeek-V4-Flash-0731-EW-MFQ) · [ModelScope](https://www.modelscope.cn/models/Tylogi/DeepSeek-V4-Flash-0731-EW-MFQ) |
| Qwen3.8-27B | `V1`–`V4`、`S4`–`S6` 档位 | [ModelScope](https://www.modelscope.cn/models/Tylogi/Qwen3.8-27B-MFQ) |
| Qwen3.6-27B | `V2`–`V3`、`S2`–`S6` 档位 | [Hugging Face](https://huggingface.co/Tylogi/Qwen3.6-27B-MFQ) |
| MiniCPM-o 4.5 | `S4`–`S8` 多模态档位 | [ModelScope](https://www.modelscope.cn/models/Tylogi/MiniCPM-o-4_5-MFQ) |

C++ 运行时目前覆盖以下架构系列。具体支持范围取决于模型版本、内嵌组件与后端。

| 架构系列 | 后端 |
| --- | --- |
| Qwen3.5–3.8 | CUDA、Metal |
| Qwen Flash-Next / Qwen4 风格 | CUDA、Metal |
| DeepSeek-V4-Flash Series | CUDA、Metal |
| MiniCPM-o 4.5 | CUDA、Metal |
| GLM5–5.3 | CUDA、Metal |
| Gemma 4 | CUDA、Metal |

部署具体模型前，请查看[运行时支持矩阵](./docs/runtime-support.md)。

## 工具与 API

| 接口 | 用途 |
| --- | --- |
| **MFQ Studio** | 本地模型目录、加载生命周期、推理 Playground、服务器状态与资源控制 |
| `mfq build` | 检测平台并编译优化的 CUDA 或 Metal C++ worker |
| `mfq quantize` | 对 HF、GGUF 或满精度 MFQ 源模型应用统一或混合精度，并写出打包 `.mfq` 模型 |
| `mfq calibrate` / `mfq solve-ew` | 收集激活与损失敏感度数据，评估并验证打包候选格式，并按精确序列化字节预算分配精度 |
| `mfq serve` | 运行 MFQ Server、Studio/Web UI、模型 worker、Cache 与持久化服务 |
| `mfq inspect` / `mfq optimize-layout` | 检查容器，并将张量重新打包为针对后端优化的布局 |

MFQ 为应用与工具提供多种接口：

- `/api/v1` 下的 **HTTP 控制 API**，用于管理模型、Session、Response、任务、
  数据集、评测、Cache 和运行时状态；
- 用于 token 与任务流式传输的 **Server-Sent Events**；
- 用于音频与全双工 Session 的 **WebSocket 实时 API**；
- 由受管 C++ worker 和运行时适配器提供的
  **OpenAI 兼容 `/v1/models` 与 `/v1/chat/completions` 接口**；
- 通过服务器工具注册表提供的 **MCP 与函数工具执行**。

可以从 [HTTP API](./docs/api/http.md)、
[WebSocket API](./docs/api/websocket.md) 或
[`mfq serve` 参考](./docs/cli/serve.md)开始。

## 性能与质量

### DeepSeek-V4-Flash-0731：模型大小与分布保真度

<p align="center">
  <img src="./docs/figures/deepseek-v4-flash-mfq-vs-ud-kld.svg" alt="DeepSeek-V4-Flash MFQ 与相近大小基线的平均 KLD 对比" width="900">
</p>

评测使用官方 0731 权重和固定 WikiText-2 协议，在 `ctx=512` 下覆盖 573 个片段
与 146,115 个计分 token。

| 发布档位 | 大小 | 平均 KLD ↓ | Top-1 一致率 ↑ |
| --- | ---: | ---: | ---: |
| `EW-V2-S` | 77.519 GiB | `0.313576` | `82.2913%` |
| `EW-V2-M` | 88.007 GiB | `0.244488` | `84.5300%` |
| `EW-V2-L` | 98.007 GiB | `0.201444` | `86.0753%` |

与本次评测中大小最接近的 Unsloth Dynamic 基线相比，MFQ 将平均 KLD 降低了
**34.24–51.42%**。

### Qwen3.5-9B：相同精度档位对比

<p align="center">
  <img src="./docs/figures/qwen35-9b-mfq-vs-ud-size-kld.svg" alt="Qwen3.5-9B MFQ 与相近大小基线的平均 KLD 对比" width="900">
</p>

图中所有档位使用同一个 BF16 教师模型，并采用覆盖完整 145 个片段、148,335 个
计分 token 的评测。MFQ 在图示每个匹配精度点上都取得了更低的原始平均 KLD。

## 文档

- [文档索引](./docs/README.md)
- [构建 C++ 运行时](./docs/cli/build.md)
- [量化模型](./docs/cli/quantize.md)
- [运行 MFQ Server](./docs/cli/serve.md)
- [运行时支持范围](./docs/runtime-support.md)
- [MiniCPM-o 4.5 多模态运行时](./docs/minicpmo45.md)
- [贡献指南与架构规范](./CONTRIBUTING.md)

## 致谢

MFQ 深受以下项目启发，并从中学习：
[llama.cpp](https://github.com/ggml-org/llama.cpp)、
[oMLX](https://github.com/jundot/omlx)、
[MLX](https://github.com/ml-explore/mlx)、
[PyTorch](https://github.com/pytorch/pytorch)、
[Transformers](https://github.com/huggingface/transformers) 和
[Unsloth](https://github.com/unslothai/unsloth)。

MFQ 使用 [Apache License 2.0](./LICENSE) 许可证。
