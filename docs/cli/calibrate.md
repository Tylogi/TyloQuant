# `mfq calibrate`

`mfq calibrate` prepares data, collects model statistics, allocates precision,
and writes artifacts for [`mfq quantize`](quantize.md). Inference does not
require calibration.

## Requirements

For a calibration-only checkout:

```shell
uv sync --extra calibration
```

Server dependencies are installed by default:

```shell
# CUDA
uv sync --extra calibration

# Apple silicon
uv sync --extra metal --extra calibration
```

## Stages

| Stage | Purpose |
|---|---|
| `data` | Tokenize eaddario calibration records |
| `trace-data` | Tokenize model-generated HF JSONL traces |
| `collect` | Collect activation and Fisher statistics |
| `imatrix` | Create activation importance on CUDA/Metal, or calibration-free uniform importance |
| `allocate` | Score candidates and allocate tensor precision |
| `candidates` | Materialize packed dense candidates without allocating a scheme |
| `inint` | Select per-neuron NINT4/NINT8 rows |
| `alphaq` | Allocate routed-expert precision from weights alone, without a calibration corpus |

Run `uv run mfq calibrate STAGE --help` for stage-specific options.

## Calibration-free AlphaQ

```shell
uv run mfq calibrate alphaq \
  --model model-hf --target-bpw 3.0 \
  --statistics alphaq-stats.json --output alphaq-3bpw.json

uv run mfq quantize model-hf model-alphaq.mfq --scheme alphaq-3bpw.json
```

No tokenizer, dataset, forward pass, gradient collection or candidate weight
fitting is required for allocation. `--device auto` uses CUDA when available,
otherwise CPU. The spectral statistics are saved after each expert bank; repeat
the command with the same `--statistics` to resume or select another budget.
Completed output schemes are never overwritten.

The built-in candidates are MFQ's standard NINT2/3/4/5/6/8 specifications.
Use repeated `--profile NINT2 --profile NINT3` to select a subset, or supply an
existing EW candidate table with `--candidates` for NVQ/native/mixed formats.
The command emits the normal scheme consumed by `mfq quantize --scheme`.
Its scope is **routed expert tensors only**; non-routed tensors follow the normal
quantizer/recipe policy, not the AlphaQ budget. The default 3.0 BPW is an example
chosen by the caller, not an AlphaQ method hyperparameter.

See [AlphaQ's formula, storage accounting and API](../quantization/alphaq.md) before comparing
budgets or using shared-codebook candidate tables.

## Activation imatrix

The imatrix stage consumes a local full-precision HF model and a prepared MFQ
calibration corpus. By default, NAQ-imatrix records two compact one-dimensional
factors for each matrix: input-channel second moments and output-neuron
importance. Layers with an activation function additionally account for that
activation when estimating neuron importance. CUDA uses FP64 accumulation by
default; Metal uses BF16 forward execution with FP32 accumulation.

### CUDA

```shell
uv run mfq calibrate imatrix \
  --model model-hf --corpus calibration-corpus \
  --output calibration.imatrix --backend cuda
```

### Metal

```shell
uv run mfq calibrate imatrix \
  --model model-hf --corpus calibration-corpus \
  --output calibration.imatrix --backend metal
```

`--device` defaults to `cuda:0` for CUDA and `mps` for Metal.
`--accumulation-dtype` overrides the backend default.
Use `--objective linear` to collect only input second moments. NAQ-imatrix keeps
one input vector and one neuron vector per dense projection, or per routed
expert; it does not store a matrix with the same shape as the weight.
For NINT tensors, the neuron factor also reallocates subgroup scale/minimum
precision at the same mean subgroup-bit budget. Loaders expand that metadata
into the existing runtime representation, so no new matmul kernel is required.

### Uniform importance without calibration

`--uniform` replaces `--corpus` and works for every architecture the HF
quantizer can plan, including architectures without an activation calibration
backend. It reads only config and safetensors headers; no tokenizer, weight
payload, corpus, forward pass or accelerator is needed.

```shell
uv run mfq calibrate imatrix \
  --model model-hf --uniform --output uniform.imatrix

uv run mfq quantize model-hf model-alphaq.mfq \
  --ud-tier IQ4_XS --imatrix uniform.imatrix --alphaq-router none
```

The artifact covers dense matrices, routed expert banks, MTP, vision and other
planned components, including fused and split layouts. Each dense input vector
has shape `(1, width)`; a rank-3 bank has shape `(matrices, width)`. Values are
all ones. Counts and token totals are zero, with objective
`uniform_calibration_free`; these are equal importance weights, not measured
activation or routing statistics. AlphaQ requires `--alphaq-router none` or a
matching measured `--alphaq-router-cache` when using this artifact.

## Reuse during quantization

Pass the artifact directly to `mfq quantize --imatrix`:

```shell
uv run mfq quantize model-bf16.gguf model-S4-L.mfq \
  --recipe quantization-recipe.gguf \
  --imatrix calibration.imatrix
```

Record the model revision, corpus, token budget, seed, attention mode, and
accumulation dtype with each run.

Full options:

```shell
uv run mfq calibrate --help
uv run mfq calibrate imatrix --help
```
