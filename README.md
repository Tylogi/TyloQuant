<div align="center">

# TyloQuant MFQ

<img src="./docs/figures/tylogi-ai-lab.svg" alt="Tylogi AI Lab" width="520">

### Next-generation quantization & inference infrastructure

**Every Bit. Maximum Fidelity.**

MFQ combines neural-network-aware SQ and VQ formats designed to approach the
practical rate-distortion frontier, high-quality, fine-grained mixed-precision
calibration and quantization, and efficient CUDA/Metal inference. The goal is
to deliver the best model quality possible on existing hardware within its
memory, storage, and latency limits.

<p>
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="Apache 2.0 license">
  <img src="https://img.shields.io/badge/runtime-C%2B%2B-black" alt="C++ runtime">
  <img src="https://img.shields.io/badge/backends-CUDA%20%7C%20Metal-6b57ff" alt="CUDA and Metal">
</p>

<p>
  <strong>English</strong> · <a href="./README.zh-CN.md">中文</a>
</p>

<p>
  <a href="#quick-start">Quick start</a> ·
  <a href="#features">Features</a> ·
  <a href="#models">Models</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="./docs/README.md">Documentation</a>
</p>

</div>

MFQ handles the full path from a source checkpoint to a deployable packed
model. It measures activation and loss sensitivity, assigns precision at
tensor, expert, and projection granularity under an exact serialized-size
budget, stores the result in a self-contained `.mfq` container, and executes
the packed weights with optimized C++ kernels. The formats and runtime are
co-designed so fidelity-per-bit gains carry through to fast CUDA/Metal
execution and one-command serving.

<p align="center">
  <img src="./docs/figures/tyloquant-mfq-webui-english.png" alt="MFQ Studio running a local model" width="900">
</p>

## Quick start

### Use a prebuilt package

The simplest Apple silicon path is a prebuilt **MFQ Studio** package. Prebuilt
packages are published through the project
[Releases](https://github.com/Tylogi/TyloQuant/releases) when available and
bundle the desktop console, MFQ Server, and C++ runtime. Open Studio,
register a model directory, and load any supported `.mfq` model.

Prebuilt C++ workers can also be supplied directly to the server:

```bash
uv run mfq serve \
  --running-executable /path/to/mfq-decode-metal \
  --model /models/model.mfq
```

### Build from source

MFQ requires Git, [uv](https://docs.astral.sh/uv/), CMake 3.26+, and a
C++ toolchain. Use CUDA 12+ on NVIDIA systems or Metal on Apple silicon.

```bash
git clone https://github.com/Tylogi/TyloQuant.git MFQ
cd MFQ

# NVIDIA / CUDA
uv sync --extra daemon
uv run mfq build --backend cuda

# Apple silicon / Metal
uv sync --extra daemon --extra metal
uv run mfq build --backend metal
```

Start an empty local server and open <http://127.0.0.1:8090/>:

```bash
uv run mfq serve
```

Detailed requirements and custom CMake options are in the
[`mfq build` guide](./docs/cli/build.md).

## Quantize and run a model

Use the MFQ **`S4-M` mixed-precision preset**. It keeps sensitive tensors at
higher precision and leaves Vision and MTP components at source precision by
default.

```bash
# Quantization dependencies
uv sync --extra train

# Hugging Face safetensors -> self-contained MFQ
uv run mfq quantize \
  /models/Qwen3.8-27B \
  /models/Qwen3.8-27B-MFQ-S4-M.mfq \
  --preset S4-M \
  --backend auto
```

Load and serve the result:

```bash
uv sync --extra daemon                 # add --extra metal on Apple silicon
uv run mfq build --backend auto
uv run mfq serve \
  --model /models/Qwen3.8-27B-MFQ-S4-M.mfq \
  --context-size 32768
```

MFQ can also quantize full-precision MFQ and GGUF sources, consume per-tensor or
Expert-Wise precision maps and activation importance matrices, and write
sharded output. See
[`mfq quantize`](./docs/cli/quantize.md) and
[`mfq calibrate`](./docs/cli/calibrate.md).

## Features

### Quantization

- **Neuron-anchored SQ and VQ base formats.** Every short subgroup in NINT, NVQ,
  NPQ, and NEPQ retains its own scale and bias. MFQ makes the common two-level
  scaling scheme more rate-efficient by storing its high-precision scale and
  bias parameters only once per neuron. At the same BPW, the saved bits can
  instead support smaller group sizes or higher-precision subgroup parameters,
  giving the base formats higher SNR and lower SSE independently of within-tensor
  precision allocation and cross-tensor bit-width assignment. On real model
  weights, they outperform mainstream alternative quantization formats in
  nearly every tested case.

- **Highest-quality calibration-free, QAT-free quantization.** MFQ incorporates
  modern, efficient methods such as DSQ to maximize the quality of one-command
  quantization without calibration data or training.

- **High-quality calibrated models.** MFQ provides high-quality, fine-grained
  mixed-precision calibration and quantization to preserve as much source-model
  quality as possible at each target size. Published calibrated models are
  evaluated against the source model under consistent model-level protocols.

- **Sub-expert and sub-tensor mixed precision.** MFQ can assign a distinct
  precision to each tensor in each expert, split a single FFN into multiple
  precision chunks, or preserve its N most important neurons at higher
  precision.
  Every choice is budgeted by its exact serialized size, and dedicated batched
  and fused kernels execute all of these heterogeneous layouts efficiently.
  This is implemented end to end for production inference, rather than
  stopping at a theoretical allocation scheme or research prototype.

- **Quantized-format backpropagation and trainers.** MFQ provides backward
  operators for its quantized formats and ready-to-use training utilities for
  convenient QAT and post-quantization model fine-tuning.

### Inference

- **High-efficiency packed-weight kernels.** Backend-specific GEMV, small-M
  MMQ, and large-M paths consume packed MFQ storage without a persistent FP16
  copy, reducing memory traffic across prefill, decode, and MTP workloads.

- **Efficient heterogeneous MoE execution.** MFQ separates the logical
  precision assigned to each expert projection from its physical kernel
  implementation. The runtime compiles and dispatches a fused path for the
  precision mix actually present, keeping heterogeneous Gate/Up/Down execution
  efficient.

- **Optimized CUDA and Metal backends.** A shared C++ model graph and runtime
  contract drive backend-specific kernels for quantized linear algebra,
  attention, convolution/recurrent state, and cache operations on NVIDIA and
  Apple hardware.

- **Broad, up-to-date architecture support.** MFQ covers diverse dense, MoE,
  multimodal, recurrent, and sparse-attention model families, and follows major
  model releases closely to support important new architectures as they appear.

- **Continuous Batching.** Dynamic request admission and active-sequence
  compaction keep concurrent generation efficient as requests arrive and
  complete, with paged KV storage and reusable execution graphs where
  supported.

- **RAM/SSD Prefix KV Cache.** Exact prefix blocks stay in a hot memory tier or
  a persistent SSD tier, allowing later requests and process restarts to reuse
  completed prefill work.

- **Composable multimodal and MTP execution.** Vision, audio, realtime duplex,
  and draft components are enabled from the model graph and share one runtime
  instead of requiring a separate serving stack for each architecture.

- **SSD-streamed inference beyond memory capacity.** A hybrid LRU/LFU cache
  keeps recently used and frequently routed experts resident, while route-aware
  prefetch coalesces cold reads across parallel I/O workers. Double-buffered
  staging and dependency-aware scheduling overlap SSD latency with useful
  compute; a bandwidth-adaptive policy splits work among resident, prefetched,
  and on-demand experts as hit rate and storage throughput change. This bounds
  RAM or VRAM use without turning each routed layer into a synchronous disk
  stall.

| Model | Model size | Expert budget | Precision | Hardware | Prefill | Decode |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| DeepSeek-V4-Flash-0731 | ~160 GiB | 85.3 GiB | Official Native QAT Precision | Apple M5 Max, 128 GB | **312.4 tok/s** | **18.6 tok/s** |

MFQ keeps heterogeneous MoE execution efficient at both supported precision
granularities:

| Model | Precision granularity | Average BPW | Prefill | Decode |
| --- | --- | ---: | ---: | ---: |
| Qwen3.8-Flash-Next | One precision per expert | **5.80** | **1.13K tok/s** | **25.4 tok/s** |
| Qwen3.8-Flash-Next | Independent precision for each expert's Gate/Up/Down | **5.79** | **1.12K tok/s** | **24.4 tok/s** |

*Warm end-to-end throughput on Apple M5 Max (40-core GPU, 128 GB).*

### What's next

MFQ is an open project that evolves with the field. We continuously explore,
validate, and integrate promising advances in quantization and inference to
deliver higher-quality quantized models through rigorous calibration and a
better local inference experience. Community contributions are warmly welcome.
See [Contributing](./CONTRIBUTING.md) to get involved.

## Models

Published `.mfq` models are available from
[Hugging Face](https://huggingface.co/Tylogi) and
[ModelScope](https://www.modelscope.cn/profile/Tylogi).

| Model | Published precision families | Download |
| --- | --- | --- |
| DeepSeek-V4-Flash-0731 | Expert-Wise `V2` tiers | [Hugging Face](https://huggingface.co/Tylogi/DeepSeek-V4-Flash-0731-EW-MFQ) · [ModelScope](https://www.modelscope.cn/models/Tylogi/DeepSeek-V4-Flash-0731-EW-MFQ) |
| Qwen3.8-27B | `V1`–`V4`, `S4`–`S6` tiers | [ModelScope](https://www.modelscope.cn/models/Tylogi/Qwen3.8-27B-MFQ) |
| Qwen3.6-27B | `V2`–`V3`, `S2`–`S6` tiers | [Hugging Face](https://huggingface.co/Tylogi/Qwen3.6-27B-MFQ) |
| MiniCPM-o 4.5 | `S4`–`S8` multimodal tiers | [ModelScope](https://www.modelscope.cn/models/Tylogi/MiniCPM-o-4_5-MFQ) |

The C++ runtime currently covers the following architecture groups. Exact
coverage depends on checkpoint revision, embedded components, and backend.

| Architecture group | Backends |
| --- | --- |
| Qwen3.5–3.8 | CUDA, Metal |
| Qwen Flash-Next / Qwen4-style | CUDA, Metal |
| DeepSeek-V4-Flash Series | CUDA, Metal |
| MiniCPM-o 4.5 | CUDA, Metal |
| GLM5–5.3 | CUDA, Metal |
| Gemma 4 | CUDA, Metal |

See the [runtime support matrix](./docs/runtime-support.md) before deploying a
specific artifact.

## Tools and APIs

| Surface | Purpose |
| --- | --- |
| **MFQ Studio** | Local model catalog, load lifecycle, inference playground, server state, and resource controls |
| `mfq build` | Detect the platform and compile the optimized CUDA or Metal C++ worker |
| `mfq quantize` | Apply uniform or mixed precision to HF, GGUF, or full-precision MFQ sources and write packed `.mfq` models |
| `mfq calibrate` / `mfq solve-ew` | Collect activation and loss-sensitivity data, score and validate packed candidates, and allocate precision to exact serialized-byte budgets |
| `mfq serve` | Run MFQ Server, Studio/Web UI, model workers, caches, and persistence |
| `mfq inspect` / `mfq optimize-layout` | Inspect containers and repack tensors into backend-optimized layouts |

MFQ exposes several interfaces for applications and tooling:

- **HTTP control API** under `/api/v1` for models, sessions, responses, jobs,
  datasets, evaluations, caches, and runtime state;
- **Server-Sent Events** for token and job streaming;
- **WebSocket realtime APIs** for audio and full-duplex sessions;
- **OpenAI-compatible `/v1/models` and `/v1/chat/completions` endpoints** from
  managed C++ workers and runtime adapters;
- **MCP and function-tool execution** through the server tool registry.

Start with the [HTTP API](./docs/api/http.md),
[WebSocket API](./docs/api/websocket.md), or
[`mfq serve` reference](./docs/cli/serve.md).

## Benchmarks

### DeepSeek-V4-Flash-0731: model size vs. distribution fidelity

<p align="center">
  <img src="./docs/figures/deepseek-v4-flash-mfq-vs-ud-kld.svg" alt="DeepSeek-V4-Flash MFQ versus matched-size baseline Mean KLD" width="900">
</p>

The evaluation uses the official 0731 weights and a fixed WikiText-2 protocol
covering 573 chunks and 146,115 scored tokens at `ctx=512`.

| Released tier | Size | Mean KLD ↓ | Same-top ↑ |
| --- | ---: | ---: | ---: |
| `EW-V2-S` | 77.519 GiB | `0.313576` | `82.2913%` |
| `EW-V2-M` | 88.007 GiB | `0.244488` | `84.5300%` |
| `EW-V2-L` | 98.007 GiB | `0.201444` | `86.0753%` |

Against the nearest-size Unsloth Dynamic baselines used in this evaluation,
MFQ lowers Mean KLD by **34.24–51.42%**.

### Qwen3.5-9B: matched precision tiers

<p align="center">
  <img src="./docs/figures/qwen35-9b-mfq-vs-ud-size-kld.svg" alt="Qwen3.5-9B MFQ versus matched-size baseline Mean KLD" width="900">
</p>

All plotted tiers use the same BF16 teacher and the complete 145-chunk,
148,335-scored-token evaluation. MFQ records lower raw Mean KLD at every
matched precision point shown.

## Documentation

- [Documentation index](./docs/README.md)
- [Build the C++ runtime](./docs/cli/build.md)
- [Quantize a model](./docs/cli/quantize.md)
- [Run MFQ Server](./docs/cli/serve.md)
- [Runtime support](./docs/runtime-support.md)
- [MiniCPM-o 4.5 multimodal runtime](./docs/minicpmo45.md)
- [Contributing and architecture rules](./CONTRIBUTING.md)

## Acknowledgements

MFQ is deeply inspired by and has learned from
[llama.cpp](https://github.com/ggml-org/llama.cpp),
[oMLX](https://github.com/jundot/omlx),
[MLX](https://github.com/ml-explore/mlx),
[PyTorch](https://github.com/pytorch/pytorch),
[Transformers](https://github.com/huggingface/transformers), and
[Unsloth](https://github.com/unslothai/unsloth).

MFQ is licensed under the [Apache License 2.0](./LICENSE).
