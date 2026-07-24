<div align="center">

# TyloQuant MFQ

<img src="./docs/figures/tylogi-ai-lab.svg" alt="Tylogi AI Lab" width="520">

**Neuron-anchored mixed-format quantization and high-fidelity LLM inference**

**Every Bit. Maximum Fidelity.**

NINT · NVQ/NPQ · NEPQ · TPQ · Gradient Precision Calibration · Expert-Wise MoE · CUDA/C++ Runtime

</div>

<p align="center">
  <strong>English</strong> | <a href="./README.zh-CN.md">中文</a>
</p>

**TyloQuant MFQ** (or **MFQ**) co-designs quantization formats, precision
allocation, and inference kernels for high-fidelity LLM deployment. It supports
custom weight encodings from `0.84-8.30 bpw`, allocates precision per compute
group or MoE expert, and executes packed weights directly through CUDA kernels
and a C++ runtime.

Public MFQ models use a unified `V`/`S` naming scheme: vector-quantized models
matched to llama.cpp `IQ*` use `V`, while scalar-quantized models matched to
`Q*_K*` use `S`. For example, `IQ3_XXS → V3-XXS` and
`Q4_K_XL → S4-L`. See [Model Naming](./MODEL_NAMING.md) for the complete rules
and registered tiers.

## Result at a Glance

### Qwen3.5-9B: disk size vs. Mean KLD

<img src="./docs/figures/qwen35-9b-mfq-vs-ud-size-kld.svg" alt="Qwen3.5-9B MFQ versus Unsloth Dynamic disk size and raw Mean KLD" width="100%">

The full WikiText-2 evaluation uses 145 chunks and 148,335 scored tokens
against the same BF16 reference. MFQ improves raw Mean KLD at every matched
precision tier shown above.

### Qwen3.6-27B: disk size vs. quality

<img src="./docs/figures/qwen36-27b-mfq-vs-ud-size-quality.png" alt="Qwen3.6-27B MFQ versus Unsloth Dynamic disk size, raw Mean KLD, and same-top accuracy" width="100%">

The aligned `ubatch=2048` evaluation compares current, matched-size MFQ files
with their Unsloth Dynamic recipes. Lower Mean KLD and higher same-top are
better. All plotted tiers use the complete 145-chunk evaluation of the current
file.

## Installation

### Requirements

- Python `>=3.10`
- Git
- An NVIDIA GPU and CUDA toolkit for CUDA inference
- CMake, Ninja, and a C++ toolchain for the native runtime

### Install from source

```powershell
git clone REPOSITORY_URL MFQ
cd MFQ
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux
source .venv/bin/activate
```

Install the quantization and calibration toolchain:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[train,calibration]"
```

Verify the installation:

```bash
mfq --help
mfq quantize --help
python -m mfq.tools.split_mfq --help
```

### Quantization

`mfq quantize` is the stable quantization entry point. It auto-detects an HF
safetensors directory or a BF16 GGUF file and calls the existing production
converter directly:

```bash
# Mixed recipe from a BF16 GGUF source
mfq quantize model-bf16.gguf model-S4-L.mfq \
  --recipe UD-Q4_K_XL.gguf --imatrix imatrix.gguf \
  --q8-mode nint8-0 --device cuda

# HF source with an expert-wise precision scheme
mfq quantize model-hf model-EW.mfq \
  --recipe UD-Q4_K_XL.gguf --ew-scheme expert-precision.json

# Important Neurons (IN); layer count is read from the recipe when possible
mfq quantize model-bf16.gguf model-IN.mfq \
  --recipe UD-Q4_K_XL.gguf --imatrix imatrix.gguf \
  --important-neurons 1024 --target-size 15G
```

The older `python -m mfq.tools.quantize_*` commands remain available for
experimental quantizer tuning. `mfq quantize` intentionally exposes only
model-level inputs, precision artifacts, sharding, and restart controls.

### Build the C++ runtime

From a Visual Studio x64 Developer PowerShell:

```powershell
cmake -S cpp_runtime -B build/cpp_runtime -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp_runtime -j 8
```

The native runtime links against a CUDA-enabled llama.cpp build. See the
[C++ runtime guide](./cpp_runtime/README.md) for `MFQ_LLAMA_BUILD_DIR`, server
startup, WebUI, and OpenAI-compatible API usage.

## WebUI

The C++ runtime includes a local WebUI at
[`http://127.0.0.1:8080/admin/`](http://127.0.0.1:8080/admin/). It provides
streaming chat, reasoning display, sampling controls, conversation history,
connection status, and runtime monitoring. The server executes each model's
embedded Jinja chat template and keeps reasoning, final content, and tool calls
as separate OpenAI-compatible channels.

<img src="./docs/figures/tyloquant-mfq-webui-english.jpg" alt="TyloQuant MFQ local inference WebUI in English" width="100%">

## How It Works

| Layer | Design | Purpose |
|---|---|---|
| Weight formats | NINT, NVQ, NPQ, NEPQ, TPQ | Quality tiers from 8 bit to below 1 bit |
| Dense allocation | Gradient precision calibration | Select precision per compute group |
| MoE allocation | Expert-Wise Precision | Spend bitrate on experts with higher expected output contribution |
| Expert container | NINTM v2 | Store heterogeneous formats in one MoE tensor |
| Runtime assets | Reserved `BLOB` records | Keep config, tokenizer, chat template, and special-token metadata in the MFQ file |
| Inference | CUDA kernels + C++ runtime | Execute packed weights without materializing full FP16 weights |

NINT shares high-level affine metadata along an output-neuron row and spends
the saved budget on shorter local groups. Below 4 bit, NVQ, NPQ, and NEPQ use
short vector codes and expert-aware codebook sharing. The allocator then chooses
formats under a real byte budget, while NINTM groups compatible experts for
direct packed execution.

See the [format specification](./FORMATS.md) for all 16 production encodings.

## Current Scope

- Qwen3.5 full attention and linear attention/GDN
- Qwen3.6 routed MoE
- Gemma4 GeGLU, sliding-window attention, and routed MoE
- DeepSeek-V4-Flash HCA/CSA/mHC and MoE
- NINTM v2 mixed-family HF/GGUF streaming conversion
- Self-contained MFQ files with embedded runtime config and GGUF tokenizer metadata
- Numbered MFQ shards with direct quantizer output and transparent Python/C++ loading
- OpenAI-compatible chat/completions API with SSE

The full-model server remains a single-GPU CUDA research prototype. Apple
silicon now has packed NINT/NVQ/NPQ/NEPQ group-vectorized GEMV, qmv_wide
small-M MMQ, online-decode `simdgroup_matrix` GEMM, and CUDA-style temporary
dequantize plus MLX GEMM at the measured large-M crossover. Compatible
NINT/VQ-family gate/up projections fuse SwiGLU. MHA/GQA/MQA, dynamic and
sliding-window KV caches, single-dispatch heterogeneous NINTM/NEPQ routing,
fused SSM/GDN kernels, GLM DSA/sparse MLA, DeepSeek-V4 compression/indexer/
sparse-attention/HC kernels, and a Qwen3.5 full/linear hybrid CausalLM
prefill/decode/generation runtime are available. Ordinary mixed-format QKV and
FFN projection groups execute in one heterogeneous Metal dispatch.
TPQ-I4G64 and TPQ-X/W/V/VV packed kernels (with legacy CCCP label
compatibility) plus the Kimi-K3 KDA/MLA,
Attention-Residual,
SiTU MoE, cache, and generation graph are also wired. DeepSeek-V4 now has
native MFQ loading, compressed/local/indexer caches, mmap-backed bounded expert
residency, prefill, decode, and generation. Gemma4 now has self-contained
sharded-MFQ loading, mixed full/sliding attention, fused norm/GeGLU/MoE,
cache, and generation. GLM-MoE-DSA has native loading, shared indexer state,
dense/sparse MLA, cache, and generation. Greedy, softmax, top-k/top-p, and
sampling-penalty kernels remain GPU-resident. Metal HTTP server integration
remains in development.
See the [Metal development status](./docs/metal.md).

## Roadmap

- [ ] Publish a reproducible Gemma-4-26B-A4B-it benchmark protocol and raw result bundle
- [ ] Add matched-size KLD and perplexity results for more Qwen3.5 and Qwen3.6 models
- [ ] Complete independent full-model KLD evaluation for DeepSeek-V4-Flash
- [ ] Re-run production decode acceptance after the NINT8 M=1 kernel fix
- [ ] Add benchmark coverage for more model families and GPU generations
- [ ] Publish downloadable quantized models with model cards and checksums
- [ ] Add continuous batching, prefix caching, and multi-GPU execution
- [ ] Add MTP speculative decoding, vision input, and structured tool calls

## Documentation

- [Benchmarks and Technical Notes](./docs/benchmarks.md)
- [Quantization Format Specification](./FORMATS.md)
- [C++ Runtime and API](./cpp_runtime/README.md)
- [Apple Silicon / Metal Development Status](./docs/metal.md)
- [MoE Observation Data Index](./plan/MoE公开Observation数据索引.md)
- [0xSero Public Resource Index](./plan/0xSero公开资源索引.md)

License: [Apache License 2.0](./LICENSE).
