# MFQ on Apple Silicon

MFQ's Apple-silicon backend uses MLX custom Metal kernels. It keeps NINT,
NVQ, NPQ, NEPQ, and TPQ weights in packed execution layouts and decodes
them inside the GPU multiply-accumulate loop; it does not retain expanded FP16
weights.

## Requirements

- Apple silicon Mac
- macOS with Metal support
- Python 3.10 or newer

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[metal]"
```

MLX JIT-compiles the kernel at first use. A separate `metal` or `metallib`
command-line compiler is not required for this path.

The mmap loader uploads the continuous NINT q-bitstream directly for
NINT2/3/4/6/8. NINT5 is repacked once into the CUDA-derived low4/high1
execution layout; this adds half a byte of padding per gs28 group but removes
general 5-bit extraction from the hot loop. No expanded FP16 weight is retained.
The much smaller group scale/min metadata is unpacked to byte-addressable
execution arrays.

NVQ, NPQ, and NEPQ preserve their packed index/state/auxiliary streams. This
includes the extended 10/12-bit NVQ2J-L/NVQ2J-XL and 10-bit NVQ3J-L codebook
profiles. Their small shared int8 codebooks are expanded into read-optimized
LUTs; the matrix weights themselves are never dequantized during loading.

TPQ product-VQ keeps 8/16-bit indices in their native element streams and
decodes p12/p14 little-endian bitstreams directly inside dense, routed,
and heterogeneous grouped kernels.

## Quantize and execute a NINT layer

```python
import mlx.core as mx
import numpy as np

from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import quantize
from mfq.runtime.mlx_linear import MlxNintLinear

weight = np.random.default_rng(0).normal(0, 0.02, (4096, 4096)).astype(np.float32)
packed = quantize(weight, NintSpec(bits=4, groupsize=24, sub_bits=6))
linear = MlxNintLinear(packed)

x = mx.random.normal((1, 4096)).astype(mx.float16)
y = linear(x)
mx.eval(y)
```

## Load an MFQ file

```python
import mlx.core as mx

from mfq.runtime.mlx_linear import MlxNintModel

model = MlxNintModel.from_mfq("model.mfq")
projection = model.linear("model.layers.0.self_attn.q_proj.weight")
y = projection(mx.zeros((1, projection.packed_weight.neuron_len)))
mx.eval(y)
```

`MlxNintModel` also provides packed `embedding()` and `ffn()` helpers for
NINT, NINT8-0, NVQ, NPQ, NEPQ, TPQ-I4G64, and TPQ product-VQ tensors.
Numbered MFQ shards are discovered transparently when any member is passed to
the loader; packed records are read from their owning shard rather than copied
into a reconstructed monolithic file.

## Run self-contained Gemma4 and GLM DSA models

Gemma4 reads its embedded Hugging Face config, mixed full/sliding layer
schedule, NINTM experts, and runtime assets directly from MFQ:

```python
import mlx.core as mx

from mfq.runtime import MlxGemma4

with MlxGemma4.from_mfq("/models/gemma4-00003-of-00008.mfq") as model:
    prompt = mx.array([[1, 42, 73]], dtype=mx.int32)
    logits = model.prefill(prompt)
    tokens = model.generate(prompt, max_new_tokens=32)
    mx.eval(logits, tokens)
```

The graph implements Gemma's BF16-rounded embedding scale, per-layer head
dimensions, partial full-attention RoPE, value normalization, delayed-softmax
router, per-expert scale, dense and routed GeGLU branches, layer scalar, final
logit soft-capping, full KV cache, and sliding-window ring cache. The
post-attention/pre-FFN norms and the final dual-branch FFN merge use dedicated
Metal kernels that preserve the CUDA runtime's intermediate FP16 rounding.

GLM-MoE-DSA uses the native HF tensor layout and a bounded context allocation:

```python
from mfq.runtime import MlxGlmDsa

with MlxGlmDsa.from_mfq("/models/glm-dsa.mfq", max_context=8192) as model:
    tokens = model.generate(prompt, max_new_tokens=32)
    mx.eval(tokens)
```

Its runtime includes low-rank Q/KV projection, adjacent-pair RoPE, NINTM
head-wise absorb/unabsorb projections, shared full-indexer state, dense MLA
prefixes, top-2048 sparse MLA, dense and sigmoid/noaux MoE FFNs, cache,
prefill, decode, and generation.

## Import and run Kimi-K3 TPQ

The importer accepts Kimi's sharded non-expert weights, converts BF16 dense
values to FP16, preserves TPQ-I4G64 matrices, and retains TPQ-X/W/V/VV
expert codebooks and indices in `NINTM` records:

```bash
python -m mfq.tools.import_tpq_to_mfq \
  --input /models/Kimi-K3-TPQ2 \
  --output /models/Kimi-K3-TPQ2.mfq
```

The manifest config and drop-expert masks are read automatically:

```python
import mlx.core as mx

from mfq.runtime import load_tpq_model

model = load_tpq_model("/models/Kimi-K3-TPQ2.mfq", device="metal")
prompt = mx.array([[1, 42, 73]], dtype=mx.int32)
logits = model(prompt)
tokens = model.generate(prompt, max_new_tokens=32)
mx.eval(logits, tokens)
```

`MlxKimiK3` implements the one-token decode graph used by TPQ2: KDA and MLA
layers, persistent short-convolution/KDA state, latent MLA cache,
Attention-Residual blocks, dense and shared SiTU MLPs, masked sigmoid routing,
latent routed experts, final residual mixing, LM head, and generation. The
current graph processes a multi-token prompt as an exact sequence of cached
decode steps; a separate Kimi prefill-batched graph remains an optimization
opportunity.

## Import and run DeepSeek-V4 TPQ

DeepSeek-V4 uses the same native MFQ import step. A raw CCCP checkpoint
directory must be converted before Metal execution:

```bash
python -m mfq.tools.import_tpq_to_mfq \
  --input /models/DeepSeek-V4-Flash-CCCP \
  --output /models/DeepSeek-V4-Flash-CCCP.mfq
```

The runtime reads the embedded manifest, assembles the HCA/CSA/mHC attention
and MoE layers, and owns all cache state:

```python
import mlx.core as mx

from mfq.runtime import load_tpq_model

model = load_tpq_model(
    "/models/DeepSeek-V4-Flash-CCCP.mfq",
    device="metal",
    cache_gb=4.0,
    max_ctx=131072,
)
prompt = mx.array([[1, 42, 73]], dtype=mx.int32)
logits = model.prefill(prompt, chunk_size=128)
tokens = model.generate(prompt, max_new_tokens=32)
mx.eval(logits, tokens)
```

The cache contains each layer's local-token ring, the compressed main pool,
and the ratio-4 indexer pool. Standard byte-aligned TPQ expert indices remain
mmap-backed on the CPU and selected experts are uploaded through a bounded
LRU controlled by `cache_gb`; no all-expert FP16 expansion is retained. Expert
rows are processed in bounded microbatches, while each selected gate/up/down
projection is still issued as one descriptor-driven Metal dispatch. Layer
boundaries are materialized so MLX lazy graphs cannot keep evicted expert
buffers alive.

## Reusable Transformer kernels

The Metal backend also provides FP16/FP32 kernels shared by common decoder
architectures:

```python
import mlx.core as mx

from mfq.kernels.metal import residual_rms_norm, rope, silu_mul
from mfq.runtime import MlxRMSNorm, MlxRoPE

norm = MlxRMSNorm(mx.ones((4096,)), eps=1e-6)
rotary = MlxRoPE(128, 32768, base=1_000_000.0)

residual, hidden = norm.add_and_forward(residual, update)
query = rotary(query, positions)
ffn_hidden = silu_mul(gate, up)
```

Available operations include RMSNorm with an optional Qwen3.5 weight offset,
L2Norm, fused residual-add plus RMSNorm, rotate-half RoPE, partial RoPE,
three-axis MRoPE, SiLU/GeLU gating, Hadamard multiplication, and residual add.
All norm reductions accumulate in FP32. GPU-resident generation operations
include stable greedy selection, full-softmax sampling, top-k/top-p sampling,
global nucleus sampling, token counts, and presence/frequency/repetition
penalties.

## Current support

| Capability | Status |
|---|---|
| NINT2/3/4/5/6/8 and NINT8-0 GEMV / MMQ / GEMM | Available |
| NINT8-1/Q8_1 activation quantize + reconstruct | Available |
| NVQ2/2J/2J-L/2J-XL/3/3J/3J-512/3J-L/1-S/1-L GEMV / MMQ / GEMM | Available |
| NPQ0-S/0-L GEMV / MMQ / GEMM | Available |
| NEPQ0/1-S/L banked GEMV / MMQ / GEMM | Available |
| TPQ-I4G64 dense matmul / dequant / embedding | Available |
| TPQ-X/W/V/VV matmul and selected-expert GEMV | Available |
| NEPQ signed-Hadamard input rotation | Available |
| NINT / NINT8-0 / NVQ / NPQ packed embedding lookup | Available |
| Heterogeneous NINT/VQ/TPQ QKV and FFN projection groups | Available, one Metal dispatch |
| Fused compatible NINT/VQ/NPQ/NEPQ gate/up + SwiGLU | Available |
| Dense F16/F32 MLX linear | Available |
| RMSNorm / L2Norm / fused residual RMSNorm | Available |
| Full / partial / three-axis rotate-half RoPE | Available |
| SiLU / GeLU gating and elementwise ops | Available |
| CPU NINT quantization | Available |
| MHA/GQA/MQA, dynamic KV cache, sliding-window cache | Available |
| GDN scalar/KDA linear attention, D=32/64/128 | Available |
| SSM conv+SiLU and fused Q/K/V conv+norm+state | Available |
| Routed NINTM mixed NINT/NINT8-0/NVQ/NPQ/NEPQ MoE | Available, single heterogeneous dispatch |
| Rotated NEPQ routed MoE | Available in the heterogeneous grouped dispatch |
| Paired routed gate/up projection | Available in one grouped dispatch |
| MoE router top-k / sqrt-softplus / weighted and shared reduction | Available as Metal kernels |
| Direct / route-compacted / expert-owned grouped MoE | Available, format- and occupancy-aware dispatch |
| GLM adjacent-pair RoPE / DSA indexer / cache write | Available |
| GLM dense 576-D K / 512-D V MLA | Available through fused MLX Metal SDPA |
| GLM selected-row sparse MLA | Available, online-softmax Metal kernel |
| DeepSeek-V4 compressor / FP4-FP8 cache / bounded decode pool delta | Available |
| DeepSeek-V4 indexer / top-512 / prefill-decode plans | Available |
| DeepSeek-V4 sparse attention with sinks / HC pre-post | Available |
| Full/linear hybrid CausalLM prefill/decode/generation | Available for Qwen3.5 layouts |
| Kimi-K3 KDA/MLA/Attention-Residual/SiTU decode graph | Available for native TPQ2 MFQ |
| DeepSeek-V4 loader/cache/prefill/decode/generation graph | Available for native TPQ MFQ |
| Gemma4 mixed full/sliding attention, GeGLU/MoE, cache and generation | Available for self-contained native MFQ |
| GLM-MoE-DSA loader, shared indexer, cache and generation | Available for self-contained native MFQ |
| Greedy / softmax / top-k / top-p / sampling penalties | Available as Metal kernels |
| Transparent numbered MFQ shard loading | Available from any shard path |
| OpenAI-compatible C++/Metal server and WebUI | Available |

The optimized schedules follow the data flow of the CUDA kernels and the
Apple-native structure used by MLX 0.32:

- GEMV consumes complete packed groups or vec4/vec8 codebook entries. NINT4
  uses its nibble-specialized one-output SIMD kernel; other NINT and VQ paths
  share activation loads across several output rows per SIMD-group.
- Small-M MMQ retains multiple input-row accumulators while reusing a decoded
  weight. VQ uses an MLX-style `qmv_wide` schedule with eight K lanes per
  output and up to five activation rows per tile.
- For `17 <= M < 64`, FP16 GEMM cooperatively expands only a transient
  `K x 64` weight tile into
  threadgroup memory. Eight SIMD-groups execute `simdgroup_matrix` multiply-
  accumulate over 32-row activation tiles. NINT8-0 uses the same schedule with
  three Q8_0 groups per K chunk, matching the data flow of CUDA's common FP16
  packed MMQ without materializing the complete weight.
- At the measured `M >= 64` crossover, the dispatcher follows the CUDA large-M
  design: dequantize the current layer to a temporary FP16 Metal array, call
  MLX dense GEMM, then release the temporary after its consumer. It is not
  retained as a second model-weight cache. Pass `dequantize_threshold=None`
  to keep the online-decode path under a strict memory budget.
- FP32 inputs retain a packed scalar fallback for correctness. Deployment and
  the optimized prefill path use FP16 activations. Accumulators are FP32.
- Routed decode concatenates all cohort bitstreams and LUTs once and stores a
  fixed-width descriptor per global expert. A single Metal dispatch reads only
  the selected expert for each token/route pair. Lanes own complete quantization
  groups, each SIMD-group shares activations across four output rows, and one
  threadgroup produces eight rows. NINT2/3/4/5/6/8, NINT8-0, and VQ vec4/vec8
  keep their specialized packed extraction inside this heterogeneous kernel.
  Signed-Hadamard activation variants for rotated NEPQ experts are generated
  once and selected by each expert descriptor. Compatible routed gate/up
  projections share the route pass and return both projections from one
  dispatch.
- Ordinary QKV and FFN projection groups use the same descriptor ABI without
  route metadata. Projections may have different output widths and may mix
  NINT2/3/4/5/6/8, NINT8-0, NVQ, NPQ, NEPQ, TPQ int4, and TPQ product-VQ
  weights in one Metal dispatch. For large M, each projection retains its
  measured temporary-dequantize plus dense-MLX crossover rather than forcing
  the grouped decode schedule past its useful range.
- TPQ2 dense int4 consumes the original low/high signed nibbles and per-row
  group-64 FP16 scales directly. TPQ expert kernels index the original shared
  FP32 codebook with u8 or u16 physical indices. The selected-expert path maps
  each global route ID to its cohort-local row and evaluates only that expert;
  it does not expand every expert or create a persistent dense copy.
- MoE dispatch uses route occupancy rather than one fixed global threshold.
  Direct packed execution is retained below roughly four routes per expert,
  route-compacted MMQ covers the middle range, and the expert-owned
  `BM=32, BN=64, BK=64` simdgroup-matrix schedule starts near sixteen routes
  per expert. Pure NINT2/3 skips the middle schedule because direct packed
  execution wins until the MMA crossover on M5. One MMA workgroup
  binary-searches the sorted range for an expert, decodes each packed weight
  tile once, applies it to every 32-route tile for that expert, accumulates K
  chunks in FP32, and writes results directly to original route positions.
  Its NINT2/3/4 loader expands eight values from one `bits`-byte packet, matching
  the useful low-bit loader idea from oMLX without adopting its older
  route-block kernel or MLX affine weight layout.
- GLM DSA uses a 32-head by 64-key by 128-dimension simdgroup-matrix indexer
  tile followed by an on-chip weighted-ReLU reduction. Its sparse MLA kernel
  assigns one threadgroup to each query head and performs online softmax over
  selected 576-D cache rows while accumulating the 512-D value prefix. Before
  the sparse threshold, 576-D keys and 512-D values are sent directly to
  MLX's fused Metal SDPA without a dense score tensor.
- Gemma4 preserves the CUDA graph's FP16 storage boundaries inside two
  one-SIMDgroup-per-row fused norm kernels. Full-attention RoPE keeps the
  physical head pairing while disabling inactive frequency pairs, rather than
  incorrectly treating partial rotary dimensions as a smaller rotate-half
  vector.
- DeepSeek-V4 uses the corresponding 64-head indexer in two 32-head MMA passes,
  plus a bounded 64-head streaming decode specialization for the first sparse
  cache band. Its exact half-key histogram top-512 kernel uses 1024 threads and
  supports deterministic lowest-index tie resolution or a faster atomic mode.
  Device-side prefill/decode plan construction feeds three sparse-attention
  schedules: short batches retain the high-grid per-head path, `M >= 32` uses
  an all-head simdgroup-matrix QK/PV path, and single-token decode uses 16
  four-head workgroups to preserve occupancy while maintaining online softmax
  and attention sinks.
  The compressor covers ordinary and overlap pooling, BF16 rounding, RoPE,
  FP4 Hadamard cache simulation, and FP8 cache simulation. HC pre/post includes
  all 20 Sinkhorn normalization iterations; the four Sinkhorn rows execute in
  parallel SIMD lanes and the collapse uses vector FMA.
- GDN follows the CUDA warp-column recurrence without storing a full
  `D x D` state in threadgroup memory. A Metal SIMD group owns one output
  column, keeps its state shard in registers across the token loop, and supports
  scalar decay, KDA per-dimension decay, contiguous/tiled GQA mapping, and
  normal or transposed cache layouts.
- The Qwen3.5 SSM path performs causal depthwise convolution, SiLU, Q/K L2
  normalization, V projection layout, and convolution-cache production in one
  dispatch. `MlxCausalLM` can assemble full-attention and linear-attention
  layers from `layer_types`.

The Metal matrix schedule was adapted from the MIT-licensed MLX Steel/QMM
structure; MFQ supplies its own NINT and VQ decoders, scale semantics, bank
selection, and execution layouts.

On an Apple M5 Max, a synthetic `4096 x 4096` NINT4 matrix occupies `9.38 MiB`
in the current Metal execution layout versus `32 MiB` for FP16. Representative
warm measurements are approximately `0.25 ms` at `M=1`, `0.23 ms` at `M=8`,
`0.42 ms` at `M=32`, and `0.49 ms` at `M=128` with automatic large-M
dispatch. The online-decode path alone is approximately `0.73 ms` at M=128.
The original scalar tiled
baseline measured about `0.79 ms` at `M=32` and `2.44 ms` at `M=128`.

A synthetic `4096 x 4080` NVQ2 matrix uses `4.09 MiB` versus `31.88 MiB` for
FP16. It measured approximately `0.26 ms` at `M=1`, `0.26 ms` at `M=8`,
`0.45 ms` at `M=32`, and `0.50 ms` at `M=128` with automatic large-M
dispatch. Its online-decode M=128 path is approximately `0.76 ms`. At `M=8`,
the packed path was
faster than the corresponding dense FP16 multiply in the same benchmark.
The benchmark reports pure dense GEMM separately from the end-to-end temporary
dequantize-plus-dense path so its crossover does not assume a persistent dense
weight.

For routed decode on the same M5 Max, an eight-expert, two-cohort NINT4/NINT5
projection with top-2 routing and `1024 x 4096` weights measured about
`0.48 ms` in the single-dispatch grouped kernel versus `0.71 ms` for the former
cohort-dispatch path (`1.49x`). This comparison is for one-token decode.
Multi-token batches can let a cohort MMQ reuse an expert weight across several
rows, so the benchmark reports those cases separately rather than claiming the
decode speedup for prefill.

For TPQ, a 32-expert mixed TPQ-X/W/V/VV projection with top-8 routing and
`256 x 512` expert weights measured `0.252 ms` in the single descriptor-driven
kernel versus `0.357 ms` for four cohort dispatches (`1.42x`). A larger
64-expert `512 x 2048` case measured `0.321 ms` versus `0.376 ms` (`1.17x`).
Both are one-token synthetic projection measurements on the same M5 Max; they
exclude router, SiTU, down projection, and full-layer overhead.

For a 256-expert, top-8 synthetic prefill projection with `512 x 2048` expert
weights on the same M5 Max, the occupancy-aware mixed NINT4/NINT5 path measured
about `2.39 ms` at 1024 routes versus `3.09 ms` for direct execution. At 4096
routes, the expert-owned MMA measured `4.50 ms`, versus `10.72 ms` direct and
`9.14 ms` route-compacted. With mixed NINT2/NINT3 weights at the same shape,
direct execution remained best at 1024 routes (`1.23 ms`), while the octet
loader plus expert-owned MMA won at 4096 routes (`2.46 ms` versus `4.16 ms`
direct). These are isolated synthetic projection timings, not end-to-end model
throughput.

For the fused linear-attention benchmark on the same M5 Max (batch 1, 8 key
heads, 16 value heads, D=128, convolution K=4), the fused SSM QKV path measured
approximately `0.38/0.29/0.32 ms` and GDN measured
`0.31/0.38/0.49 ms` at `T=1/32/128`. These are isolated kernel-dispatch
measurements, not complete model-layer latency.

For the common-operation benchmark on the same machine (`4096` hidden width,
32 heads, 128 head dimensions), decode-sized dispatches measured approximately
`0.27 ms` for RMSNorm, `0.22 ms` for fused residual plus RMSNorm, `0.16 ms` for
SiLU gating, and `0.16 ms` for RoPE. The measurements include Python and MLX
evaluation overhead.

For isolated DeepSeek-V4 sparse attention on the same M5 Max (`B=1`, `H=64`,
`D=512`, cache length 768, 640 selected rows), the four-head decode schedule
measured approximately `0.99 ms` versus `1.88 ms` for the former per-head
threadgroup (`1.91x`). At `M=64`, the all-head simdgroup-matrix prefill path
measured `3.10 ms` versus `6.56 ms` (`2.11x`). It was slower for `M <= 16`,
which is why automatic dispatch retains the old high-grid schedule below
`M=32`. For a 16K-row top-k input, deterministic tie handling measured
approximately `0.43 ms`; atomic bucketed output measured `0.27 ms`.

## Remaining CUDA parity gaps

Functional coverage and one-to-one kernel scheduling are tracked separately.
The main remaining gaps are:

- DeepSeek-V4 now has native MFQ loading, layer assembly, bounded cache
  lifetime management, prefill, decode, and generation. It has synthetic
  official-shape kernel/graph coverage, but still needs an independently
  reproducible released-checkpoint KLD and end-to-end throughput acceptance
  run. MTP/DSpark speculative decoding and the Metal HTTP server are not wired.
- GLM now has a native loader, shared-indexer state, dense/sparse MLA, cache,
  prefill, decode, and generation runtime. It still needs a released-checkpoint
  KLD and end-to-end throughput acceptance run.
- Kimi-K3 has a complete decode/generation graph, but has not yet received a
  released full-model KLD/throughput acceptance run. Prompt ingestion currently
  replays the decode graph token by token instead of using a dedicated batched
  Kimi prefill schedule.
- The graph-safe `dsv4_decode_pool_step` API returns only bounded
  remainder/history state, one possible compressed row, and its destination
  index. It never makes the long-context pool a custom-kernel output.
  `dsv4_decode_pool_update` remains as a capacity-backed compatibility wrapper
  and applies that delta with an indexed MLX update. A model runtime should
  call `mx.eval` after cache updates to detach prior decode graphs.
- Full/sliding dense attention is functionally available through fused MLX
  Metal SDPA, and sampling has dedicated Metal kernels. CUDA's explicit
  split-K decode, planned SWA, fully fused penalty-and-sample path, and every
  model-specific dense-attention schedule do not yet have one-to-one Metal
  kernels.
- The fused NINT LM-head argmax, projection-input gate epilogues, and fully
  fused gate/up/down FFN variants remain unfused on Metal. Their unfused Metal
  compositions are functionally equivalent.
- The expert-owned grouped MMA substantially improves direct heterogeneous
  prefill, but generic multi-family decode remains slightly behind separate
  homogeneous cohort dispatch for some large, balanced route distributions.
- GLM sparse MLA and short DSV4 prefill still use portable per-head
  online-softmax schedules. DSV4 large prefill now has simdgroup-matrix QK/PV,
  but CUDA's occupancy-driven stream-K/fixup schedule is not structurally
  portable and can remain faster at very large selected counts.
