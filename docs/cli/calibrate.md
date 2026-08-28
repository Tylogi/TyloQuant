# `mfq calibrate`

`mfq calibrate` prepares data, collects model statistics, allocates precision,
and writes artifacts for [`mfq quantize`](quantize.md). Inference does not
require calibration.

## Requirements

For a calibration-only checkout:

```shell
uv sync --extra calibration
```

A checkout that also serves models needs both sets of extras:

```shell
# CUDA
uv sync --extra daemon --extra calibration

# Apple silicon
uv sync --extra daemon --extra metal --extra calibration
```

## Stages

| Stage | Purpose |
|---|---|
| `data` | Tokenize eaddario calibration records |
| `trace-data` | Tokenize model-generated HF JSONL traces |
| `collect` | Collect activation and Fisher statistics |
| `imatrix` | Create a reusable activation importance matrix on CUDA or Metal |
| `allocate` | Score candidates and allocate tensor precision |
| `candidates` | Materialize packed dense candidates without allocating a scheme |
| `inint` | Select per-neuron NINT4/NINT8 rows |

Run `uv run mfq calibrate STAGE --help` for stage-specific options.

## Activation imatrix

The imatrix stage consumes a local full-precision HF model and a prepared MFQ
calibration corpus. CUDA uses FP64 accumulation by default; Metal uses BF16
forward execution with FP32 accumulation.

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
