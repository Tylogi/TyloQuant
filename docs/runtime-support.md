# Runtime support

MFQ runtime support is experimental. Coverage depends on the model revision and
quantization recipe.

## Coverage terms

- **End-to-end**: native MFQ loading, prefill, decode, and generation are wired.
- **Partial**: architecture-specific conversion, graph, or kernel components
  exist, but there is no complete public model runtime.
- **Dedicated**: the model uses a separate TPQ/MFQ execution path.

## Model and platform coverage

| Model or family | Conversion and packaging | Native CUDA | Metal | Scope or current limitation |
|---|---|---|---|---|
| Qwen3.5 | HF/GGUF/full-precision MFQ | End-to-end | End-to-end | Full/linear hybrid CausalLM |
| Qwen3.6 | HF/GGUF/full-precision MFQ | Partial | Partial | Routed-MoE components; no public end-to-end runtime |
| Gemma4 | HF/GGUF and sharded MFQ | End-to-end | End-to-end | Mixed full/sliding attention |
| DeepSeek-V4-Flash | HF/GGUF, Expert-Wise MFQ, and TPQ | End-to-end | End-to-end | Compression, indexer, and sparse-attention paths |
| MiniCPM-o 4.5 | Official composite HF graph to MFQ | End-to-end | End-to-end | Official model directory required at runtime |
| GLM-MoE-DSA | Native MFQ and mixed precision | End-to-end | End-to-end | Dense/sparse MLA paths |
| Kimi-K3 | TPQ/MFQ packaging | Dedicated | Dedicated | TPQ/MFQ execution path only |

Test each artifact on its target backend before deployment.

## Shared conversion, storage, and serving

- NINTM v2 mixed-family HF/GGUF streaming conversion.
- Self-contained MFQ files with embedded runtime configuration, tokenizer,
  chat template, special-token metadata, and optional sampling profiles.
- Numbered MFQ shards with direct quantizer output and Python/C++ loading.
- OpenAI-compatible chat and completions APIs with server-sent events.

Related docs: [self-contained releases](release.md),
[`mfq serve`](cli/serve.md), and the [HTTP API](api/http.md).

## CUDA status

The default CUDA path is a native C++/CUDA runtime with no Python, PyTorch,
ATen, or LibTorch dependency at execution time. It currently runs on one GPU.
The [native CUDA validation plan](cuda-native-runtime-validation.md) covers the
optional migration A/B runtime.

## Apple silicon and Metal

### Packed compute

- Packed NINT/NVQ/NPQ/NEPQ group-vectorized GEMV.
- `qmv_wide` small-M MMQ and online-decode `simdgroup_matrix` GEMM.
- Temporary dequantization plus MLX GEMM beyond the measured large-M
  crossover.
- TPQ-I4G64, TPQ-X/W/V/VV, and three-projection TPQ-P kernels with p8-p16
  indices.

### Fused and heterogeneous execution

- Fused SwiGLU for compatible NINT/VQ-family gate/up projections.
- Single-dispatch heterogeneous NINTM/NEPQ routing.
- Mixed-format QKV and FFN projection groups in one heterogeneous Metal
  dispatch.
- GPU-resident greedy, softmax, top-k/top-p, and sampling-penalty kernels.

### Attention and state

- MHA/GQA/MQA with dynamic and sliding-window KV caches.
- Fused SSM/GDN kernels and GLM DSA/sparse MLA.
- DeepSeek-V4 compression, indexer, sparse-attention, and HC kernels.
- Kimi-K3 KDA/MLA, Attention-Residual, SiTU MoE, cache, and generation graph.

## End-to-end runtime details

- **Qwen3.5:** full/linear hybrid CausalLM prefill, decode, and generation.
- **DeepSeek-V4:** native MFQ loading; compressed, local, and indexer caches;
  mmap-backed bounded expert residency; prefill; decode; and generation.
- **Gemma4:** self-contained sharded-MFQ loading; mixed full/sliding attention;
  fused norm/GeGLU/MoE; cache; and generation.
- **GLM-MoE-DSA:** native loading, shared indexer state, dense/sparse MLA,
  cache, and generation.
- **MiniCPM-o 4.5:** text, image, video, audio, and duplex workflows are covered
  in the [runtime guide](minicpmo45.md).

## Partial and dedicated paths

- **Qwen3.6:** conversion and routed-MoE graph/kernel components; no public
  end-to-end runtime.
- **Kimi-K3:** KDA/MLA, Attention-Residual, SiTU MoE, cache, and generation on
  the TPQ/MFQ path.

Build requirements and backend selection: [`mfq build`](cli/build.md).
