# `mfq quantize`

`mfq quantize` converts an HF safetensors directory, a full-precision MFQ, or a
full-precision GGUF. It detects the source format unless `--source-format` is
set.

## Requirements

Quantization checks free space before tensor fitting and again before final
assembly. Requirements are added per filesystem: staged/fitted payloads and the
final output coexist by default, even when output is sharded. `--alphaq-cache`,
`--temp-dir` and output can use separate disks. Estimates include selected vision
and MTP payloads, assets, and 5% plus 64 MiB headroom per filesystem for variable
codec tables, metadata and allocation overhead. Free space is a snapshot; other
writers and quotas can still cause a later write to fail. Imatrix collection uses
the same check for its BF16 hidden-state workspace and output arrays.

For AlphaQ, `--alphaq-consume-fitted` releases each fitted tensor after copying it
into the final output. On one filesystem, peak use approaches the larger of cache
and output plus one tensor and metadata. This discards fitted-cache resume:
interruption requires refitting deleted tensors, and successful export also
consumes them. Statistics and trained tables remain reusable. The option rejects
shared `encoded-candidates.json` caches; it only consumes individual fitted blobs.

The `train` extra installs PyTorch and safetensors. For a quantization-only
checkout:

```shell
uv sync --extra train
```

Server dependencies are installed by default:

```shell
# CUDA
uv sync --extra train

# Apple silicon
uv sync --extra metal --extra train
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

Vision encoders and MTP/draft predictors remain at their source precision by
default, independently of the text quantization policy. Quantize either scope
only when that tradeoff is intentional:

```shell
uv run mfq quantize model-hf model-NINT4-all.mfq \
  --bits 4 --quantize-vision --quantize-mtp
```

Built-in standard mixtures are available without supplying a recipe GGUF:

```shell
uv run mfq quantize model-hf model-S4-M.mfq --preset S4-M
```

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
in the [Expert-Wise joint budget solver](../quantization/ew-joint-solver.md).

## AlphaQ quantization

On an HF source, `--target-bpw` runs the calibration-free AlphaQ allocator
over the whole file, including PLE tables, native weights, headers, and
assets:

```shell
uv run mfq quantize model-hf model-alphaq.mfq --target-bpw 3.0
```

`--alphaq-dense` selects how non-routed tensors join the budget (`auto` uses
the evaluated UD recipe for Qwen3.8 Flash Next, otherwise joint allocation).
`--alphaq-profiles` narrows the candidate set, `--alphaq-router` and
`--alphaq-router-cache` control MoE router importance, `--devices` pins CUDA
indices for parallel statistics and training, and `--alphaq-cache` selects a
reusable statistics and trained-tables directory. The allocation method is
documented in [AlphaQ](../quantization/alphaq.md).

`--ud-tier IQ1_S` selects a registered model's pinned nonexpert recipe and
exact UD main-text byte ceiling; it cannot combine with `--target-bpw`.
Select `--alphaq-allocator natscale` for [NatScale](../quantization/natscale.md).
Its sensitivity defaults to `uniform`, tau to 1.0 and errors to `measured`;
run `mfq calibrate natscale-errors` first. `--natscale-error analytic` selects
the explicit bit-decay ablation. NatScale requires recipe dense mode.
In these whole-file workflows, `--alphaq-vision {native,NINT8,exclude}` and
`--alphaq-mtp {quantize,exclude}` default to native vision and quantized MTP
when present, with both payloads reported outside the main-text ceiling.

## Format and scope options

| Option | Meaning |
|---|---|
| `--nvq3-jsc`, `--nvq3-jsc-512`, `--nvq3-to-nint3`, `--iq2-s-to-nint2`, `--npq0-l` | Opt eligible tensors into the named packed format. |
| `--nint-data-free` | Allocate per-neuron NINT profiles from reconstruction SSE without calibration data. |
| `--nvq-calibration {auto,none,gain,group24}` | NVQ codebook calibration source. |
| `--dense-dtype {f16,f32}` | HF dtype for recipe tensors left unquantized. |
| `--ple-dtype {recipe,native,NINT4,NINT8}` | AlphaQ PLE lookup precision; `recipe` defaults to native when no recipe is available. |
| `--text-only` | Omit non-language tensors. |
| `--exclude-mtp` | Omit recipe-only MTP blocks. |
| `--tokenizer GGUF`, `--model-config JSON` | Override tokenizer/chat-template metadata and the model config source. |
| `--tensor-overrides JSON` | Map exact GGUF tensor names to final MFQ dtypes. |
| `--temp-dir PATH` | HF intermediate tensor workspace. |
| `--reuse-unweighted-from MFQ` | GGUF: reuse raw blobs for tensors without an imatrix binding. |

NVQ search and codebook training knobs (`--nvq-*`, `--npq0-l-*`,
`--nvq3-jsc-banks`, `--nvq3-jsc-learned-scale`) trade quality against speed;
the defaults are calibrated, and `uv run mfq quantize --help` lists every one.

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

HF and full-precision-MFQ quantization use the low-disk streaming writer by
default. Each completed tensor blob is appended to an output-side private
payload and deleted immediately, so peak workspace is approximately one model
copy plus the tensor currently being encoded. Use `--staged-blobs` to select
the older two-phase path that retains the complete blob set until final
assembly.

- `--split-max-size N[M|G]` limits tensor payload per output shard.
- `--split-max-tensors N` limits the tensor count per shard.
- `--staged-blobs` retains all HF/MFQ tensor blobs before final assembly.
- `--resume` reuses validated HF or Important-Neuron temporary blobs; for
  HF/MFQ inputs it implies `--staged-blobs`.
- `--resume-completed N` reuses completed, validated GGUF tensor blobs.
- `--dry-run` validates and plans without writing final output.
- `--overwrite` replaces an existing output.
- `--keep-temp` retains temporary blobs for diagnosis or restart and likewise
  implies `--staged-blobs` for HF/MFQ inputs.

The shard limits are mutually exclusive. Python and native runtimes load
numbered shards as one model family.

## Calibration and metadata

[`mfq calibrate`](calibrate.md) creates reusable activation imatrices.
`--sampling-profile` embeds a versioned
[runtime sampling profile](../runtime/sampling-profiles.md).

Full options: `uv run mfq quantize --help`.
