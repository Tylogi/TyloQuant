"""Public, source-agnostic MFQ quantization command.

The low-level converters remain importable for experiments.  This module is
the stable user-facing surface: it normalizes common options and then calls
the same converter functions without adding another quantization path.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mfq.formats.shards import parse_size

QUANT_BACKENDS = ("auto", "cuda", "metal", "cpu")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "quantize",
        help="convert or quantize HF, GGUF, and full-precision MFQ models",
        description=(
            "Quantize an HF safetensors directory, a full-precision MFQ, "
            "or a full-precision GGUF. "
            "A GGUF recipe defines mixed tensor precision; --scheme adds "
            "per-tensor or expert-wise (EW) overrides."
        ),
    )
    parser.add_argument("input", help="HF checkpoint directory, GGUF, or MFQ file")
    parser.add_argument("output", help="output .mfq path")
    source = parser.add_argument_group("source")
    precision = parser.add_argument_group("precision and metadata")
    important = parser.add_argument_group("important neurons (IN)")
    execution = parser.add_argument_group("execution")
    output = parser.add_argument_group("output and restart")
    source.add_argument(
        "--source-format",
        choices=("auto", "hf", "gguf", "mfq"),
        default="auto",
        help="source layout; auto detects HF directories and .gguf/.mfq files",
    )
    precision.add_argument(
        "--recipe",
        default="",
        help="GGUF whose tensor types define the mixed-precision recipe",
    )
    precision.add_argument(
        "--scheme",
        "--ew-scheme",
        dest="scheme",
        default="",
        help="prepared MFQ tensor/expert precision scheme (supports EW)",
    )
    precision.add_argument(
        "--imatrix",
        default="",
        help="native MFQ, GGUF, or legacy llama.cpp importance matrix",
    )
    precision.add_argument(
        "--tokenizer",
        default="",
        help="GGUF providing tokenizer, chat template, and special-token metadata",
    )
    precision.add_argument(
        "--model-config",
        default="",
        help="model config JSON; defaults beside the source checkpoint",
    )
    precision.add_argument(
        "--sampling-profile",
        default="",
        help="versioned runtime sampling profile JSON to embed in the output MFQ",
    )
    precision.add_argument(
        "--tensor-overrides",
        default="",
        help="JSON mapping exact GGUF tensor names to final MFQ dtypes",
    )
    precision.add_argument("--bits", type=int, default=4, help="HF uniform NINT bit width")
    precision.add_argument("--groupsize", type=int, default=24, help="HF uniform NINT group size")
    precision.add_argument("--sub-bits", type=int, default=6, help="HF NINT scale precision")
    precision.add_argument(
        "--q8-mode",
        choices=("nint8", "nint8-0"),
        default="nint8",
        help="mapping for recipe Q8_0 tensors",
    )
    precision.add_argument(
        "--nvq-calibration",
        choices=("auto", "none", "gain", "group24"),
        default="auto",
    )
    precision.add_argument("--nvq3-jsc", action="store_true")
    precision.add_argument("--nvq3-jsc-512", action="store_true")
    precision.add_argument("--nvq3-to-nint3", action="store_true")
    precision.add_argument("--iq2-s-to-nint2", action="store_true")
    precision.add_argument("--npq0-l", action="store_true")
    precision.add_argument(
        "--nvq-codebook-scope",
        choices=("fixed", "tensor"),
        default="tensor",
    )
    precision.add_argument("--nvq-codebook-artifact-dir", default="")
    precision.add_argument("--nvq-jsc-row-importance", default="")
    precision.add_argument(
        "--dense-dtype",
        choices=("f16", "f32"),
        default="f32",
        help="HF dtype for recipe tensors left unquantized",
    )
    precision.add_argument(
        "--full-precision",
        "--bf16",
        dest="bf16",
        action="store_true",
        help=(
            "copy HF native BF16, block-FP8, and MXFP4 tensors into a "
            "full-precision MFQ without MFQ quantization (--bf16 is a legacy alias)"
        ),
    )
    source.add_argument("--text-only", action="store_true", help="omit non-language tensors")
    source.add_argument("--exclude-mtp", action="store_true", help="omit recipe-only MTP blocks")
    important.add_argument(
        "--important-neurons",
        "--in-top-k",
        dest="important_neurons",
        type=int,
        default=0,
        metavar="TOP_K",
        help="split the TOP_K imatrix-ranked dense FFN neurons into a higher-precision branch",
    )
    important.add_argument(
        "--in-baseline",
        default="",
        help="existing base MFQ for IN; otherwise a temporary base model is produced",
    )
    important.add_argument(
        "--in-layers",
        type=int,
        default=0,
        help="model layer count for IN; inferred from recipe metadata when omitted",
    )
    important.add_argument(
        "--in-layer-indices",
        default="",
        help="comma-separated layer subset for IN; defaults to all layers",
    )
    important.add_argument(
        "--target-size",
        type=parse_size,
        default=0,
        metavar="N[M|G]",
        help="IN output byte budget; defaults to recipe file size",
    )
    execution.add_argument(
        "--row-chunk",
        type=int,
        default=0,
        help="streaming rows per quantization batch; 0 selects a backend-specific value",
    )
    execution.add_argument("--nvq-group-chunk", type=int, default=1024)
    execution.add_argument("--nvq-search-steps", type=int, default=19)
    execution.add_argument("--nvq-assignment", choices=("native", "torch"), default="native")
    execution.add_argument("--nvq-jsc-banks", type=int, choices=(1, 2, 4), default=4)
    execution.add_argument("--nvq-jsc-iterations", type=int, default=4)
    execution.add_argument("--nvq-jsc-assignment-refine-steps", type=int, default=2)
    execution.add_argument("--nvq-jsc-raw-multiplier", type=int, default=8)
    execution.add_argument("--nvq3-jsc-banks", type=int, choices=(1, 2, 4), default=2)
    execution.add_argument("--nvq3-jsc-learned-scale", action="store_true")
    execution.add_argument("--npq0-l-iterations", type=int, default=4)
    execution.add_argument("--npq0-l-assignment-refine-steps", type=int, default=2)
    execution.add_argument("--npq0-l-fixed-refine-steps", type=int, default=3)
    execution.add_argument("--npq0-l-kmeans-iterations", type=int, default=8)
    execution.add_argument("--npq0-l-group-chunk", type=int, default=512)
    execution.add_argument("--nvq1-l-candidates", type=int, default=0)
    execution.add_argument(
        "--nvq1-l-anchor-multipliers",
        type=float,
        nargs="+",
        default=(0.75,),
    )
    execution.add_argument("--nvq1-l-refine-steps", type=int, default=2)
    execution.add_argument("--nvq1-l-assignment", choices=("native", "torch"), default="native")
    execution.add_argument("--nvq-codebook-train-rows", type=int, default=2048)
    execution.add_argument("--nvq-codebook-validation-rows", type=int, default=512)
    execution.add_argument("--nvq-codebook-row-chunk", type=int, default=512)
    execution.add_argument("--nvq-codebook-iterations", type=int, default=4)
    execution.add_argument("--nvq-codebook-projection-candidates", type=int, default=48)
    execution.add_argument("--nvq-codebook-min-improvement", type=float, default=0.0)
    execution.add_argument("--nvq-codebook-seed", type=int, default=20260716)
    execution.add_argument("--backend", choices=QUANT_BACKENDS, default="auto")
    execution.add_argument("--device", default="cuda")
    split = output.add_mutually_exclusive_group()
    split.add_argument(
        "--split-max-size",
        type=parse_size,
        default=0,
        metavar="N[M|G]",
        help="maximum tensor payload per MFQ shard",
    )
    split.add_argument(
        "--split-max-tensors",
        type=int,
        default=0,
        help="maximum tensor count per MFQ shard",
    )
    output.add_argument("--temp-dir", default="", help="HF temporary blob directory")
    output.add_argument(
        "--resume",
        action="store_true",
        help="resume size-validated HF or IN temporary blobs",
    )
    output.add_argument(
        "--resume-completed",
        type=int,
        default=0,
        metavar="N",
        help="GGUF: reuse the first N completed, validated tensor blobs",
    )
    output.add_argument(
        "--reuse-unweighted-from",
        default="",
        help="GGUF: reuse raw blobs for tensors without an imatrix binding",
    )
    output.add_argument("--dry-run", action="store_true")
    output.add_argument("--overwrite", action="store_true")
    output.add_argument("--keep-temp", action="store_true")
    parser.set_defaults(_impl=run)


def _detect_source(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if path.is_dir():
        if (path / "model.safetensors.index.json").is_file() or any(path.glob("*.safetensors")):
            return "hf"
        raise ValueError(f"cannot detect an HF safetensors checkpoint under {path}")
    if path.is_file() and path.suffix.lower() == ".gguf":
        return "gguf"
    if path.is_file() and (
        path.suffix.lower() == ".mfq" or re.search(r"-[0-9]{5}-of-[0-9]{5}\.mfq$", path.name)
    ):
        return "mfq"
    raise ValueError(f"cannot detect source format for {path}")


def _append_value(argv: list[str], option: str, value: object) -> None:
    if value not in (None, "", 0, False):
        argv.extend((option, str(value)))


def _append_flag(argv: list[str], option: str, enabled: bool) -> None:
    if enabled:
        argv.append(option)


def _append_vq_arguments(argv: list[str], args: argparse.Namespace) -> None:
    for option, value in (
        ("--nvq-calibration", args.nvq_calibration),
        ("--nvq-codebook-scope", args.nvq_codebook_scope),
        ("--nvq-codebook-artifact-dir", args.nvq_codebook_artifact_dir),
        ("--nvq-jsc-row-importance", args.nvq_jsc_row_importance),
        ("--nvq-group-chunk", args.nvq_group_chunk),
        ("--nvq-search-steps", args.nvq_search_steps),
        ("--nvq-assignment", args.nvq_assignment),
        ("--nvq-jsc-banks", args.nvq_jsc_banks),
        ("--nvq-jsc-iterations", args.nvq_jsc_iterations),
        (
            "--nvq-jsc-assignment-refine-steps",
            args.nvq_jsc_assignment_refine_steps,
        ),
        ("--nvq-jsc-raw-multiplier", args.nvq_jsc_raw_multiplier),
        ("--nvq3-jsc-banks", args.nvq3_jsc_banks),
        ("--npq0-l-iterations", args.npq0_l_iterations),
        (
            "--npq0-l-assignment-refine-steps",
            args.npq0_l_assignment_refine_steps,
        ),
        ("--npq0-l-fixed-refine-steps", args.npq0_l_fixed_refine_steps),
        ("--npq0-l-kmeans-iterations", args.npq0_l_kmeans_iterations),
        ("--npq0-l-group-chunk", args.npq0_l_group_chunk),
        ("--nvq1-l-candidates", args.nvq1_l_candidates),
        ("--nvq1-l-refine-steps", args.nvq1_l_refine_steps),
        ("--nvq1-l-assignment", args.nvq1_l_assignment),
        ("--nvq-codebook-train-rows", args.nvq_codebook_train_rows),
        ("--nvq-codebook-validation-rows", args.nvq_codebook_validation_rows),
        ("--nvq-codebook-row-chunk", args.nvq_codebook_row_chunk),
        ("--nvq-codebook-iterations", args.nvq_codebook_iterations),
        (
            "--nvq-codebook-projection-candidates",
            args.nvq_codebook_projection_candidates,
        ),
        ("--nvq-codebook-min-improvement", args.nvq_codebook_min_improvement),
        ("--nvq-codebook-seed", args.nvq_codebook_seed),
    ):
        _append_value(argv, option, value)
    if args.nvq1_l_anchor_multipliers:
        argv.append("--nvq1-l-anchor-multipliers")
        argv.extend(str(value) for value in args.nvq1_l_anchor_multipliers)
    for option, enabled in (
        ("--nvq3-jsc", args.nvq3_jsc),
        ("--nvq3-jsc-512", args.nvq3_jsc_512),
        ("--nvq3-to-nint3", args.nvq3_to_nint3),
        ("--iq2-s-to-nint2", args.iq2_s_to_nint2),
        ("--npq0-l", args.npq0_l),
        ("--nvq3-jsc-learned-scale", args.nvq3_jsc_learned_scale),
    ):
        _append_flag(argv, option, enabled)


def _infer_in_layers(recipe: Path) -> int:
    from mfq.tools.quantize_gguf_to_mfq import _field_value, _load_gguf

    gguf_reader_class, _dequantize = _load_gguf()
    reader = gguf_reader_class(str(recipe), "r")
    architecture = _field_value(reader, "general.architecture", "")
    value = _field_value(reader, f"{architecture}.block_count", None)
    if not isinstance(value, int) or value <= 0:
        raise ValueError("cannot infer IN layer count; pass --in-layers")
    return value


def _gguf_arguments(args: argparse.Namespace, output: Path) -> argparse.Namespace:
    from mfq.tools.quantize_gguf_to_mfq import build_parser

    argv = [
        "--input-bf16-gguf",
        str(Path(args.input).resolve()),
        "--recipe-gguf",
        str(Path(args.recipe).resolve()),
        "--output",
        str(output),
        "--row-chunk",
        str(args.row_chunk),
        "--quant-backend",
        args.backend,
        "--device",
        args.device,
    ]
    _append_value(argv, "--tokenizer-gguf", args.tokenizer)
    _append_value(argv, "--model-config", args.model_config)
    _append_value(argv, "--sampling-profile", args.sampling_profile)
    _append_value(argv, "--calibration-scheme", args.scheme)
    _append_value(argv, "--tensor-precision-overrides", args.tensor_overrides)
    _append_value(argv, "--imatrix", args.imatrix)
    _append_value(argv, "--resume-completed", args.resume_completed)
    _append_value(argv, "--reuse-unweighted-from", args.reuse_unweighted_from)
    _append_value(argv, "--split-max-size", args.split_max_size)
    _append_value(argv, "--split-max-tensors", args.split_max_tensors)
    _append_flag(argv, "--exclude-mtp", args.exclude_mtp)
    _append_flag(argv, "--q8-to-nint8-zero", args.q8_mode == "nint8-0")
    _append_vq_arguments(argv, args)
    _append_flag(argv, "--dry-run", args.dry_run)
    _append_flag(argv, "--overwrite", args.overwrite)
    _append_flag(argv, "--keep-temp", args.keep_temp)
    return build_parser().parse_args(argv)


def _hf_arguments(
    args: argparse.Namespace,
    output: Path,
    *,
    input_option: str = "--input",
) -> argparse.Namespace:
    from mfq.tools.quantize_hf_to_mfq import build_parser

    argv = [
        input_option,
        str(Path(args.input).resolve()),
        "--output",
        str(output),
        "--bits",
        str(args.bits),
        "--groupsize",
        str(args.groupsize),
        "--sub-bits",
        str(args.sub_bits),
        "--row-chunk",
        str(args.row_chunk),
        "--quant-backend",
        args.backend,
        "--device",
        args.device,
        "--dense-dtype",
        args.dense_dtype,
    ]
    _append_value(argv, "--recipe-gguf", args.recipe)
    _append_value(argv, "--tokenizer-gguf", args.tokenizer)
    _append_value(argv, "--model-config", args.model_config)
    _append_value(argv, "--sampling-profile", args.sampling_profile)
    _append_value(argv, "--calibration-scheme", args.scheme)
    _append_value(argv, "--tensor-precision-overrides", args.tensor_overrides)
    _append_value(argv, "--imatrix", args.imatrix)
    _append_value(argv, "--split-max-size", args.split_max_size)
    _append_value(argv, "--split-max-tensors", args.split_max_tensors)
    _append_value(argv, "--temp-dir", args.temp_dir)
    _append_flag(argv, "--text-only", args.text_only)
    _append_flag(argv, "--bf16", args.bf16)
    _append_flag(argv, "--q8-to-nint8-zero", args.q8_mode == "nint8-0")
    _append_vq_arguments(argv, args)
    _append_flag(argv, "--resume-temp", args.resume)
    _append_flag(argv, "--dry-run", args.dry_run)
    _append_flag(argv, "--overwrite", args.overwrite)
    _append_flag(argv, "--keep-temp", args.keep_temp)
    return build_parser().parse_args(argv)


def _mfq_arguments(args: argparse.Namespace, output: Path) -> argparse.Namespace:
    return _hf_arguments(args, output, input_option="--input-mfq")


def _full_precision_arguments(
    args: argparse.Namespace,
    output: Path,
) -> argparse.Namespace:
    from mfq.tools.convert_hf_to_full_mfq import build_parser

    argv = [
        "--input",
        str(Path(args.input).resolve()),
        "--output",
        str(output),
    ]
    _append_value(argv, "--tokenizer-gguf", args.tokenizer)
    _append_value(argv, "--model-config", args.model_config)
    _append_value(argv, "--sampling-profile", args.sampling_profile)
    _append_value(argv, "--split-max-size", args.split_max_size)
    _append_value(argv, "--split-max-tensors", args.split_max_tensors)
    _append_value(argv, "--temp-dir", args.temp_dir)
    _append_flag(argv, "--resume", args.resume)
    _append_flag(argv, "--dry-run", args.dry_run)
    _append_flag(argv, "--overwrite", args.overwrite)
    _append_flag(argv, "--keep-temp", args.keep_temp)
    return build_parser().parse_args(argv)


def _run_in(args: argparse.Namespace, baseline: Path, output: Path) -> None:
    from mfq.tools.quantize_important_neurons import build_parser, convert

    recipe = Path(args.recipe).resolve()
    layers = args.in_layers or _infer_in_layers(recipe)
    argv = [
        "--input-bf16-gguf",
        str(Path(args.input).resolve()),
        "--recipe-gguf",
        str(recipe),
        "--baseline-mfq",
        str(baseline),
        "--imatrix",
        str(Path(args.imatrix).resolve()),
        "--output",
        str(output),
        "--layers",
        str(layers),
        "--top-k",
        str(args.important_neurons),
        "--row-chunk",
        str(args.row_chunk),
        "--device",
        args.device,
    ]
    _append_value(argv, "--layer-indices", args.in_layer_indices)
    _append_value(argv, "--target-bytes", args.target_size)
    _append_flag(argv, "--dry-run", args.dry_run)
    _append_flag(argv, "--resume", args.resume)
    _append_flag(argv, "--overwrite", args.overwrite)
    _append_flag(argv, "--keep-temp", args.keep_temp)
    convert(build_parser().parse_args(argv))


def _validate(args: argparse.Namespace, source_format: str) -> None:
    if args.bf16 and source_format != "hf":
        raise ValueError("--full-precision requires an HF safetensors source")
    if args.bf16 and (args.recipe or args.scheme or args.imatrix):
        raise ValueError(
            "--full-precision cannot be combined with --recipe, --scheme, or --imatrix"
        )
    if args.bf16 and args.tensor_overrides:
        raise ValueError("--tensor-overrides do not apply to --full-precision")
    if args.bf16 and args.important_neurons:
        raise ValueError("--full-precision cannot be combined with important-neuron quantization")
    if args.bf16 and (args.bits, args.groupsize, args.sub_bits) != (4, 24, 6):
        raise ValueError("--bits, --groupsize, and --sub-bits do not apply to --full-precision")
    if args.bf16 and args.dense_dtype != "f32":
        raise ValueError("--dense-dtype does not apply to --full-precision")
    if source_format == "gguf" and not args.recipe:
        raise ValueError("GGUF quantization requires --recipe")
    if source_format == "gguf" and args.text_only:
        raise ValueError("--text-only is only valid for an HF source")
    if source_format == "gguf" and (args.bits != 4 or args.groupsize != 24 or args.sub_bits != 6):
        raise ValueError("--bits/--groupsize/--sub-bits are only valid for an HF source")
    if source_format == "gguf" and args.dense_dtype != "f32":
        raise ValueError("--dense-dtype is only valid for an HF source")
    if source_format == "gguf" and args.temp_dir:
        raise ValueError("--temp-dir is only valid for an HF source")
    if source_format == "gguf" and args.resume and not args.important_neurons:
        raise ValueError("GGUF resume requires --resume-completed N")
    if source_format in {"hf", "mfq"} and args.exclude_mtp:
        raise ValueError("--exclude-mtp currently requires a GGUF source")
    if args.resume_completed and source_format != "gguf":
        raise ValueError("--resume-completed is only valid for GGUF quantization")
    if args.important_neurons < 0:
        raise ValueError("--important-neurons must be non-negative")
    if args.important_neurons:
        if source_format != "gguf":
            raise ValueError("IN quantization requires a BF16 GGUF source")
        if not args.imatrix:
            raise ValueError("IN quantization requires --imatrix")
        if args.split_max_size or args.split_max_tensors:
            raise ValueError("IN output splitting is not supported by the current writer")
        if args.dry_run and not args.in_baseline:
            raise ValueError("IN --dry-run requires --in-baseline")
    elif args.in_baseline or args.in_layers or args.in_layer_indices or args.target_size:
        raise ValueError("IN-only options require --important-neurons TOP_K")


def run(args: argparse.Namespace) -> int:
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    source_format = _detect_source(source, args.source_format)
    _validate(args, source_format)
    print(
        json.dumps(
            {
                "event": "mfq_quantize",
                "source_format": source_format,
                "input": str(source),
                "output": str(output),
                "recipe": str(Path(args.recipe).resolve()) if args.recipe else None,
                "scheme": str(Path(args.scheme).resolve()) if args.scheme else None,
                "imatrix": str(Path(args.imatrix).resolve()) if args.imatrix else None,
                "full_precision": args.bf16,
                "important_neurons": args.important_neurons or None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if source_format == "hf":
        if args.bf16:
            from mfq.tools.convert_hf_to_full_mfq import convert

            convert(_full_precision_arguments(args, output))
            return 0
        from mfq.tools.quantize_hf_to_mfq import convert

        convert(_hf_arguments(args, output))
        return 0

    if source_format == "mfq":
        from mfq.tools.quantize_hf_to_mfq import convert

        convert(_mfq_arguments(args, output))
        return 0

    from mfq.tools.quantize_gguf_to_mfq import convert

    if not args.important_neurons:
        convert(_gguf_arguments(args, output))
        return 0

    owned_baseline = not bool(args.in_baseline)
    baseline = (
        Path(args.in_baseline).resolve()
        if args.in_baseline
        else output.parent / f".{output.name}.in-baseline.mfq"
    )
    if owned_baseline and baseline.exists() and not args.resume:
        raise FileExistsError(
            f"temporary IN baseline exists; pass --resume or remove it: {baseline}"
        )
    if owned_baseline and not baseline.is_file():
        base_args = _gguf_arguments(args, baseline)
        base_args.split_max_size = 0
        base_args.split_max_tensors = 0
        base_args.dry_run = False
        convert(base_args)
    _run_in(args, baseline, output)
    if owned_baseline and not args.keep_temp and baseline.is_file():
        baseline.unlink()
    return 0


__all__ = ["add_parser", "run"]
