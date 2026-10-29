# TQ

TQ uses a fixed procedural Gaussian codebook and independent 1–8 bit precision
for each output row. It applies no rotation. Each 256-weight ring uses a
16-bit Trellis state, with FP16 row scales and either 4-bit subgroup gains or
shared FP16 input-channel scales. The binary payload remains `RTRELL05`.

The CUDA encoder uses two-pass Viterbi search and supports diagonal imatrix
weighting. GEMV and SwiGLU FFN dispatch precision per row on the GPU, without
sorting or grouping rows. Inference reconstructs FP16 weights, accumulates in
FP32, and accepts FP16 or BF16 activations. FFN uses two kernels, including
intermediate activation rounding and memory traffic.

The default search fits 1–8 bit rows using shared column scales, sampled scale
selection, complete source-range coverage and sparse row-scale repair. It
retains the full 16-bit Trellis state space and protects FP16 path costs from
overflow. Legacy subgroup fitting uses `column_scaling=False` with
`search="adaptive"` or `search="full"`; the full search uses the original FP32
encoder and exhaustive scale trials. All paths produce the same TQ format.

```python
import numpy as np
import torch

from mfq.formats.tq import TQTensor
from mfq.kernels.cuda.tq import Projection
from mfq.quantize.tq import quantize

weight = torch.randn(2048, 4096, device="cuda")
bits = (np.arange(2048) % 8 + 1).astype(np.uint8)
importance = torch.ones(4096, device="cuda")
tensor = quantize(weight, bits, imatrix=importance, groupsize=512)
tensor.save("projection.tq")

loaded = TQTensor.load("projection.tq")
projection = Projection(loaded, input_columns=weight.shape[1])
x = torch.randn(4096, device="cuda", dtype=torch.float16)
y = projection.gemv(x)
```

`SwiGLU(gate, up)` fuses Gate, Up and the activation.
`SwiGLUFFN(gate, up, down)` adds Down projection. Preallocated outputs and
workspace support allocation-free CUDA graph capture.

The Metal API is `mfq.kernels.metal.tq.Projection`. It consumes the same payload,
supports ordinary and routed projections, and fuses independent Gate/Up tensors
without precision sorting. Native Metal loading accepts the `TQ` record dtype;
`tq_shapes` in the MFQ header records logical `[output, input]` or
`[experts, output, input]` dimensions. Stored rows are expert-major. TQ execution
does not pass through MFE. The Metal expert cache for Flash-Next and DeepSeek-V4.1 keeps the three
TQ banks in shared compressed byte pools, with per-expert offsets and LRU
eviction. Expert reads use vectored file I/O; repeated routes share one load.
Small streamed batches with sufficient expert overlap also reuse decoded rows
across up to four routed inputs. This uses route IDs already available for I/O;
the fully resident path does not add a host synchronization to construct a plan.
Changing an active cache budget rebuilds only its compressed expert pools;
dense weights and request KV state remain loaded. Invalid budgets leave the
current pools unchanged. Resident TQ experts can switch to the same streaming
cache when the weight budget decreases, without replacing the model or request KV.
Pool allocations never exceed the complete compressed expert payload plus its
runtime metadata and alignment, even when the configured budget is larger.
For subgroup sizes of at least 256, Metal prepares FP32 group-scale metadata at
load time; encoded weights and their FP16 reconstruction remain unchanged.
Each MoE layer stores three TQ tensors: Gate, Up and Down, with logical
`[experts, output, input]` dimensions and independent row bit widths, scales and
codes. Gate/Up fusion reads the two original tensors together without repacking
them or creating precision cohorts. Ordinary projection batches share one Metal dispatch across
independent TQ tensors, including different output widths and scale layouts.
On M5, batched projections decode directly into NAX fragments; routed prefill
uses a TQ row-block plan to reuse each expert's weights across tokens.
Long dense prefill can decode once into a temporary matrix before GEMM,
with a 256 MiB staging limit. FP16 activation paths use FP16 temporary weights;
FP32 and BF16 paths promote the exact reconstructed FP16 values to FP32.
This does not require NAX, and the expanded matrix is not cached as a model
weight. Longer routed batches share decoded tiles in threadgroup memory.
Without NAX, sufficiently populated routed batches use SIMD-group matrix
multiply with the same compressed tiles and row-block plan.
Narrow-output, long-input projections split the input dimension into FP32
partials before their final reduction. These dispatches depend on geometry,
not row precision or model identity.
Short-input routed Down projections fuse the weighted expert reduction; longer
inputs keep the independently measured projection and reduction path.
PLE and Engram row stores gather the selected TQ rows without expanding their
complete tables. Selected rows with different subgroup sizes or column scales
share one decode dispatch; metadata is prepared only for those rows.
Tables larger than one payload can use ordered row shards without changing the
TQ binary format. The MFQ header's `tq_row_shards` object maps each logical table
name to an ordered array of physical TQ record names. These records concatenate
along the row axis and share the logical input width; each keeps its own row
precisions and scale layout. Physical names append `.tq_row_shard.00000`,
`.tq_row_shard.00001`, and so on to the logical name. The logical name must not
also be a physical record. Row stores resolve and cache only the requested rows
across shards; they do not concatenate or expand the full table in memory.

Tests: `pytest tests/test_formats/test_tq.py tests/test_tq_cuda.py tests/test_tq_metal.py`.

## ROCm kernels

`mfq.kernels.hip.tq.Projection` operates on the same `RTRELL05` payload on
MI300X. Small-token calls use the CUDA TQ vector decoder ported to 64-lane AMD
waves, including normalized bitstream windows, packed FP16 book arithmetic,
static 1..8-bit row dispatch and FP32 accumulation. Single-token geometry avoids
unused token accumulators; aligned FP32 inputs use complete vector loads. Register
barriers preserve intermediate rounding without volatile private-memory traffic.
Larger batches select compressed tiles or temporary decoding plus rocBLAS GEMM
according to the matrix geometry, for both forward and input gradients. The temporary
matrix is released after each call. It uses two bytes per weight for FP16 inputs
and four for BF16/FP32 inputs, in addition to GEMM workspace and activation buffers.
BF16 inputs retain exact FP16 weight values through FP32 GEMM or, for sufficiently
large batches, two exact BF16 weight components in one expanded BF16 GEMM.
`split_bf16=True` explicitly selects the latter. FP16 GEMM
returns FP32 results before bias and final output rounding.

`Projection(..., staging=False)` selects compressed computation for all batch
sizes. This path shares decoded values across tokens without materializing a full
weight matrix. Both paths support FP16, BF16 and FP32 activations and preserve
the existing first-order input-gradient API.

```python
from mfq.kernels.hip.tq import Projection

projection = Projection(tensor, input_columns=weight.shape[1])
x = torch.randn(17, weight.shape[1], device="cuda", requires_grad=True)
y = projection(x)
y.square().mean().backward()
```

PyTorch uses the `cuda` device name on ROCm. Autograd computes first-order input gradients
through the fixed compressed matrix; codes, row anchors, subgroup selectors,
column scales and bias remain fixed. `projection.backward(dy, out=dx)` runs
the input-gradient kernel directly. `projection.forward(x, out=y)` and the
explicit backward method accept preallocated buffers for graph capture.
The HIP `SwiGLU` and `SwiGLUFFN` helpers compose these projections with
PyTorch activations, including their input gradients.

`mfq.quantize.tq.quantize` selects HIP fitting kernels on a ROCm PyTorch
build. Both the full FP32 Viterbi encoder and adaptive paired-FP16 search
retain the full Trellis state space. Adaptive 1-bit search keeps all but 512
path costs in LDS, leaving room for block metadata. Adaptive 2-bit search stages
all next costs in registers before replacing its LDS table. Adaptive 2?4-bit
tracebacks store two predecessor choices per byte. These changes preserve the
CUDA fitting algorithm, scale trials, normalization and serialized output.
The full FP32 reference encoder continues to use its separate scratch layout.

The native HIP runtime also loads ordinary two-dimensional TQ projections with
the same payload and logical-shape metadata. It uses the compressed small-token
decoder, combined projection groups and fused SwiGLU where supported. CPU
offload and tensor parallelism for TQ are rejected. Routed-expert TQ caches are
not implemented by this port.

Native inference offers `MFQ_HIP_TQ_Q8=1` for CUDA's Q8 activation algorithm on
compatible small-token FP16/FP32 calls. It quantizes activations per 16 values and
uses the affine-book approximation from CUDA. This is disabled by default and
is not used by the Python calibration API. Its additional projection error and
model quality require separate assessment. `MFQ_HIP_TQ_TRACE=1` reports actual
single-token activation types and dispatch roles once per role/type combination.

Tests: `pytest tests/test_tq_hip.py tests/test_tq_hip_production.py tests/test_tq_hip_fused.py tests/test_tq_hip_q8.py tests/test_formats/test_tq.py`.
CUDA-only PTX tests require CUDA hardware rather than a ROCm installation.


### Quantization measurements and multiple GPUs

`bench/tq_hip_bench.py` fits real 0.8B/4B Qwen projection weights and retains each
TQ payload. `--reference` requires byte-identical results from an earlier run.
It measures fitting separately from calibration forward/backward GPU time.

`bench/tq_hip_checkpoint.py` distributes independent text-layer 2D projections
across the requested devices. Each projection has an atomic TQ payload and a
source/payload hash record for restart. No complete model container is assembled.
Every requested projection must succeed before the run reports completion;
optional progress-file or stdout failures do not stop fitting. Resumed work is
reported separately from newly fitted parameter throughput.

```bash
python bench/tq_hip_checkpoint.py \
  --source /workspace/hf/Qwen3.5-0.8B \
  --output /workspace/mfq/.mfq/tq-checkpoint \
  --gpus 0 1 2 3 4 5 6 7 --bits 4
```

This checkpoint helper defaults to unweighted adaptive fitting. The tensor API accepts
`imatrix` for calibration-weighted fitting. Each matrix retains ordinary TQ
storage; GPU scheduling does not add expert axes or change the wire format.


On 2026-10-24, MI300X measurements from `TQ@9d52748` with ROCm PyTorch 2.12
used real Qwen projection weights. A 3584x1024 projection fitted at 1 bit in
0.916 s versus 2.417 s before, and at 2 bits in 0.327 s versus
0.842 s before. The complete 186-projection, 497,614,848-parameter 0.8B
TQ4 job took 96.66 s on one GPU and 14.58 s on eight GPUs with the
same checkpoint helper. All outputs matched the earlier quantizer byte for byte.
The complete checkpoint boundary includes fitting, reference checks and payload
writes; it excludes process startup and full model-container assembly. No stable
single-GPU 4-bit fitting gain was established, and these are not cross-hardware
CUDA/Metal comparisons.

### Production column fitting on HIP

The tensor API now defaults to `search="fast"`, `column_scaling=True` and
`repair_scales=True`, following upstream TQ `6d08119`. Row scales cover the
complete source range. Each 256-value ring rescales its path metric by a power
of two when needed to prevent FP16 path-cost overflow. Sparse scale repair is
accepted only when actual FP16 weighted reconstruction error decreases.

`mfq.quantize.cuda.tq_columns` dispatches to the existing optimized HIP column
encoder on ROCm. `ColumnFitter` retains shared preprocessing across bit widths;
the historical imports from `tq_fitter` refer to this same implementation.
Legacy subgroup fitting requires explicit `column_scaling=False` and
`search="adaptive"` or `search="full"`. The timing records below predate the
new zero-clipping search policy unless explicitly labeled otherwise.

Task-generated models belong under `/workspace/mfq/models`, which maps to the
SSH user's `~/mfq/models`. Existing source models keep their
original locations. The speed fixture is
`/workspace/mfq/models/qwen35-4b-tq-mixed-speed.mfq`.

The 5090 Qwen3.5-9B production run used a retained CUDA fitter beyond the
`TQ@9d52748` branch snapshot. Its actual path is now available on HIP:

```python
tensor = quantize(weight, bits, imatrix=importance,
                  search="fast", column_scaling=True)
```

This path supports all 1..8-bit profiles with complete 16-bit states. It uses
16 sampled rows, one scale-screening pass, one full-matrix fit, 32 boundary
warmup steps followed by 256 encoding steps, and a persistent native kernel.
The existing adaptive path remains available with its original search settings.
Column fitting uses the existing RTRELL05 column-scale representation; it does
not introduce another container or expert axis. Shorter state spaces are not
enabled by this port.

The HIP kernel keeps all 1-bit costs in 64 KiB LDS and uses the existing global
scratch allocation for that block's metadata. Two-bit costs are replaced in
place. Paired traceback matches the production CUDA algorithm for 2/3-bit
profiles. Packed FP16 arithmetic and integer score/choice keys preserve the
first candidate on a tie. Candidate comparisons at 6/7/8 bits are distributed
across groups of 2/4/8 lanes, with an exact subgroup minimum. Every profile
caches its fixed codebook points in registers. Cache indices are static after
unrolling so the compiler does not place these arrays in thread-private memory.

For all eight profiles, captured native inputs from the 5090 production fitter
gave exactly identical HIP states, boundary indices and decoded values across
16 cases of 1024 real rings. The fixture is a correctness check, not a throughput
estimate. The full `[4096,4096]` Qwen3.5-9B `blk.0.attn_gate.weight` comparison
used the original BF16 weights and imatrix:

| Bits | MI300X initial port seconds | MI300X optimized seconds | Retained 5090 seconds |
|---:|---:|---:|---:|
| 1 | 0.758 | 0.695 | 0.648 |
| 2 | 0.350 | 0.263 | 0.242 |
| 3 | 0.292 | 0.229 | 0.221 |
| 4 | 0.263 | 0.202 | 0.222 |
| 5 | 0.262 | 0.201 | 0.320 |
| 6 | 0.425 | 0.220 | 0.332 |
| 7 | 0.437 | 0.237 | 0.313 |
| 8 | 0.468 | 0.191 | 0.290 |

Candidate time includes fitting, reconstruction/error calculation, packing and
payload/error writes; it excludes source loading, shared column preparation,
compilation and complete model assembly. The 5090 figures are retained timings
from its five-worker production run; they are not a new controlled single-GPU
timing sweep. Full payloads are byte-identical for 5..8 bits. For 1..4 bits,
FP32 preprocessing/reduction differences produce at most two changed anchors
and 93 changed packed bytes; aggregate weighted SSE differs by less than
`8e-8` relatively. Native kernels are exact when supplied the same inputs.
This does not constitute full-model WT2 quality acceptance.

The checkpoint runner can fit every profile after one weight upload and one
column preparation per matrix, retaining an independent checkpoint per profile:

```bash
python bench/tq_hip_checkpoint.py \
  --source /workspace/hf/Qwen3.5-0.8B \
  --output /workspace/mfq/.mfq/tq-columns-fast \
  --gpus 0 1 2 3 4 5 6 7 --bits 1 2 3 4 5 6 7 8 \
  --search fast --column-scaling
```

The checkpoint helper uses uniform importance; weighted production parity is
checked by `bench/tq_hip_production.py`. Run the HIP regression tests with
`pytest tests/test_tq_hip_production.py tests/test_tq_hip.py tests/test_formats/test_tq.py`.

The eight-GPU Qwen3.5-0.8B run completed all 186 projections at all eight
profiles (1488 payloads, 3,980,918,784 fitted parameter instances) in 11.029 s
versus 18.645 s before this kernel optimization. Time includes column preparation,
fitting, packing, source/output hashes and writes. The optimized run also reads
and compares every old payload; all 1488 files are byte-identical. It reported
no failures and fitted every candidate anew. This run uses uniform importance
and does not predict Qwen3.5-9B end-to-end time. Resume, optional progress/stdout
failures, and required compute-failure propagation were checked separately.

Full-matrix native encoding times on the same 16M-weight projection changed
from 393/325/268/243/239/396/394/435 ms at 1..8 bits to
299/241/205/178/181/199/202/168 ms. These native intervals exclude Python
preprocessing and output writing. End-to-end outputs remain unchanged from
the HIP production port. The initial production regression run passed 126 tests, including
all profiles, padded/mixed row widths, zero-importance columns and zero rows.

On gfx942 the measured thread counts are 1024 for 1..3/5 bits, 256 for 4/8
bits and 512 for 6/7 bits. The 4/5/8-bit variants permit two resident blocks per
CU rather than one. The loader also canonicalizes equivalent fast-library
arguments to avoid loading the same JIT library twice. Hardware counter
collection timed out on this host; no measured peak-utilization percentage is
claimed. The final kernel uses vector FP16 operations, not MFMA instructions.

The independent-process launcher `bench/tq_hip_multiprocess.py` uses one worker
per requested GPU and reuses the per-projection checkpoint protocol. A complete
eight-GPU Qwen3.5-4B 1..8-bit run produced 1,984 payloads in 57.065 s. These are
fresh fits with `search="fast"`, `column_scaling=True`, full 16-bit states and
unchanged search iterations. They do not constitute a measured 9B run.

`bench/tq_hip_model.py` assembles retained fitted rows at per-projection byte
budgets for native speed testing. Its deterministic row assignment is not a
validated precision-allocation recipe. `bench/tq_hip_model_bench.py` checks model
and binary hashes, prompt/completion counts and saved timing samples. Model
quality acceptance remains separate from these runtime measurements.

The native HIP runtime loads ordinary RTRELL05 projections without an expert
axis. Small token batches decode the mixed row profiles directly; larger
batches materialize the exact weights for matrix multiplication. FP16 dot pairs
and FP32/FP16 mixed multiply instructions preserve the weight reconstruction and
input activation types. Grouped projections, SwiGLU, residual addition, input
gates and the single-token GDN normalization path reduce intermediate kernels.
Optional activation Q8 remains disabled by default and was not used below.

On one MI300X (GPU 4), the retained Qwen3.5-4B mixed-row speed fixture gives:

| Input tokens | Native TQ decode tokens/s | llama.cpp Q4_K_M decode tokens/s |
|---:|---:|---:|
| 1 / llama-bench depth 0 | 244.75 | 243.58 |
| 2048 | 241.89 | 241.82 |
| 8192 | 231.61 | 234.10 |

Each TQ row is the median of three measured samples after one warmup, with all
256 requested completion tokens verified, a 16384-token context and MTP disabled.
Only synthetic `benchmark_prompt_tokens` requests continue across EOS; ordinary
requests retain their existing EOS and stop handling. The checkpoint includes
model and binary SHA-256 hashes and rejects incomplete generation. Earlier
attention measurements that stopped at EOS after 12 tokens are invalid.

The llama.cpp reference uses commit `31288c9c6`, the same GPU, FP16 KV, three
repetitions and `llama-bench -p 2048 -n 256 -d 0,2048,8192 -ngl 99`. The two
runtimes use different synthetic prompts, harnesses and weight formats. Active
ordinary projection byte budgets are approximately matched; the TQ fixture has
not passed quality acceptance. These numbers establish runtime speed, not equal
model quality. At 2048 input tokens, native TQ prefill is 8999 tokens/s versus
11701 for llama.cpp; prefill remains slower.

Both attention implementations use tiled softmax and context partitioning.
The TQ FP16 decode kernel uses packed dot pairs with FP32 accumulation and a
parallel segment reducer; its grouped-head KV reuse differs from llama.cpp's
decode tile kernel. Before the attention changes, the 8192-input trace spent
865 microseconds per token in attention, including its reducer. More context
partitions and the parallel reducer reduced that to 361 microseconds before the
subsequent packed-dot and reducer-unrolling changes. Prefill uses the existing
MFMA attention implementation. Trace timings include profiling overhead and are
not substituted for the end-to-end timing samples above.

Validation passed 280 Python tests and the native HIP TQ/decode, inference and
text-processor CTests. Native checks include independent FP64 attention oracles,
exact weight reconstruction and exact typed normalization/gate comparisons.
Attention accumulation order can change floating-point results and generated
token sequences; byte identity is not claimed for complete model outputs.

Long-context measurements on GPU 4 use 16384/32768/65536 input tokens and
256 verified output tokens. Medians of three measured samples are:

| Input tokens | TQ decode tokens/s | llama.cpp decode tokens/s | TQ prefill seconds | llama.cpp prefill seconds |
|---:|---:|---:|---:|---:|
| 16384 | 228.22 | 225.23 | 1.857 | 1.392 |
| 32768 | 211.28 | 216.94 | 3.940 | 2.996 |
| 65536 | 185.03 | 200.00 | 8.439 | 6.854 |

These TQ measurements explicitly use `--prefill-chunk-size 65536` and
`--ctx-size 65792`. This is a runtime configuration, not a change to the default
2048-token prefill chunk. The first prefill chunk can use the MFMA attention
path; later chunks currently use the generic continuation-attention path.
Default-chunk prefill measured 11.36 seconds at 16K and about 42.15 seconds at
32K. Those completed samples are retained, but that superseded run did not
complete its 64K cases. Whole-prompt prefill avoids that slow continuation path
on this MI300X. A dedicated MFMA implementation for cached continuation
prefill remains separate work.

The llama.cpp reference retains its default batch size 2048 and microbatch size
512, with separate full-input PP and depth-adjusted TG tests. Both runtimes use
FP16 KV, disable MTP, and retain the same weights used in the shorter tests.
Optional TQ activation Q8 remains disabled. Benchmark records include binary
and model hashes. The speed-fixture and quality limitations above still apply.
