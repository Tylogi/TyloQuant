# `mfq quantize`

`mfq quantize` converts an HF safetensors directory, a full-precision MFQ, or a
full-precision GGUF. It detects the source format unless `--source-format` is
set.

## Requirements

The `train` extra installs PyTorch and safetensors. For a quantization-only
checkout:

```shell
uv sync --extra train
```

A checkout that also serves models needs both sets of extras:

```shell
# CUDA
uv sync --extra daemon --extra train

# Apple silicon
uv sync --extra daemon --extra metal --extra train
```

## Source modes

| Input | Main options | Result |
|---|---|---|
| HF directory | `--bits`, `--groupsize`, `--sub-bits` | Uniform NINT or recipe-driven MFQ |
| HF directory | `--full-precision` | Exact self-contained full-precision MFQ |
| Full-precision MFQ | Precision options or a recipe | Quantized MFQ |
| Full-precision GGUF | `--recipe RECIPE.gguf` | Mixed-format MFQ |

GGUF quantization requires a recipe. A prepared `--scheme` (also accepted as
`--ew-scheme`) can add per-tensor or Expert-Wise overrides. Use `--imatrix` to
supply a native MFQ, GGUF, or legacy llama.cpp importance matrix.

## Basic NINT quantization

Quantize an HF checkpoint directly with a uniform NINT configuration:

```shell
uv run mfq quantize model-hf model-NINT4.mfq \
  --bits 4 --groupsize 24 --sub-bits 6 --backend auto
```

`--backend auto` prefers CUDA, then Metal, then CPU. Pass an explicit backend
to pin the execution path.

## Full-precision MFQ

Copy native HF storage without MFQ quantization. BF16 remains BF16; supported
block-FP8/MXFP4 values and their E8M0 scales remain self-contained and exact.

```shell
uv run mfq quantize model-hf model-full.mfq --full-precision
```

Quantize that file with:

```shell
uv run mfq quantize model-full.mfq model-NINT3.mfq \
  --bits 3 --groupsize 24 --sub-bits 5 --backend cpu --device cpu
```

## Mixed GGUF recipe

```shell
uv run mfq quantize model-bf16.gguf model-S4-L.mfq \
  --recipe quantization-recipe.gguf --imatrix imatrix.gguf \
  --q8-mode nint8-0 --device cuda
```

## Expert-Wise overrides

```shell
uv run mfq quantize model-hf model-EW.mfq \
  --recipe quantization-recipe.gguf --ew-scheme expert-precision.json
```

A scheme may contain per-tensor or per-expert decisions. Allocation is covered
in the [Expert-Wise joint budget solver](../ew-joint-solver.md).

## Add an MTP head

Add an MTP head from the original HF checkpoint to an existing quantized
model. Backbone blobs stay byte-identical. Each MTP decoder projection uses the
precision of the corresponding final backbone layer.

```shell
uv run mfq quantize model-hf model-with-MTP.mfq \
  --base-mfq model-quantized.mfq --backend metal --device mps
```

`--base-mfq` cannot be combined with a recipe, scheme, imatrix, tensor
overrides, or `--full-precision`.

## Important Neurons

Split the highest-ranked dense FFN neurons into a higher-precision branch:

```shell
uv run mfq quantize model-bf16.gguf model-IN.mfq \
  --recipe quantization-recipe.gguf --imatrix imatrix.gguf \
  --important-neurons 1024 --target-size 15G
```

MFQ reads the layer count from recipe metadata. If it is missing, pass
`--in-layers`. Use `--in-layer-indices` for a layer subset.

## Sharding and restart controls

- `--split-max-size N[M|G]` limits tensor payload per output shard.
- `--split-max-tensors N` limits the tensor count per shard.
- `--resume` reuses validated HF or Important-Neuron temporary blobs.
- `--resume-completed N` reuses completed, validated GGUF tensor blobs.
- `--dry-run` validates and plans without writing final output.
- `--overwrite` replaces an existing output.
- `--keep-temp` retains temporary blobs for diagnosis or restart.

The shard limits are mutually exclusive. Python and native runtimes load
numbered shards as one model family.

## Calibration and metadata

[`mfq calibrate`](calibrate.md) creates reusable activation imatrices.
`--sampling-profile` embeds a versioned
[runtime sampling profile](../runtime-sampling-profiles.md).

Full options: `uv run mfq quantize --help`.
