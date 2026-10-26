"""MFQ command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from mfq._version import __version__


def _not_implemented(args: argparse.Namespace) -> int:
    print(f"command {args.command!r} is not implemented", file=sys.stderr)
    return 2


def _voice_runtime_check(_args: argparse.Namespace) -> int:
    import onnxruntime  # noqa: F401
    import s3tokenizer  # noqa: F401
    import scipy  # noqa: F401
    import soundfile  # noqa: F401
    import torch  # noqa: F401
    import torchaudio  # noqa: F401
    from hyperpyyaml import load_hyperpyyaml  # noqa: F401
    from stepaudio2.flashcosyvoice.modules.hifigan import HiFTGenerator  # noqa: F401
    from stepaudio2.token2wav import _setup_cosyvoice2_alias  # noqa: F401

    print(json.dumps({"voice_output_runtime": "ready"}))
    return 0


def _flash_next_worker(args: argparse.Namespace) -> int:
    from mfq.runtime.flash_next_worker import run_worker

    return run_worker(args)


def _mlx_runtime_check(_args: argparse.Namespace) -> int:
    """Touch the bundled MLX Metal runtime without loading a model."""
    import mlx.core as mx

    value = mx.array([1.0], dtype=mx.float32)
    mx.eval(value)
    print(json.dumps({"mlx_metal_runtime": "ready"}))
    return 0


def _hub_download(args: argparse.Namespace) -> int:
    """Download a hub snapshot from the unified CLI used by packaged Studio builds."""
    if args.provider == "huggingface":
        from huggingface_hub import snapshot_download

        output = snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            local_dir=args.local_dir,
            allow_patterns=args.include or None,
            ignore_patterns=args.exclude or None,
            max_workers=args.max_workers,
            token=os.environ.get("HF_TOKEN") or None,
        )
    else:
        from modelscope_hub import HubApi

        api = HubApi(
            token=os.environ.get("MODELSCOPE_API_TOKEN")
            or os.environ.get("MODELSCOPE_TOKEN")
            or None
        )
        output = api.download_repo(
            args.repo_id,
            args.repo_type,
            revision=args.revision,
            local_dir=args.local_dir,
            allow_patterns=args.include or None,
            ignore_patterns=args.exclude or None,
            max_workers=args.max_workers,
        )
    print(json.dumps({"event": "hub_download_complete", "path": str(output)}))
    return 0


def _calibrate_data(args: argparse.Namespace) -> int:
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("calibration corpus generation requires transformers") from exc
    from mfq.calibration.dataset import build_eaddario_corpus, eaddario_sources

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )
    corpus = build_eaddario_corpus(
        tokenizer,
        args.output,
        repo_id=args.repo,
        sources=eaddario_sources(args.source_size),
        cache_dir=args.cache_dir or None,
        train_tokens=args.train_tokens,
        validation_tokens=args.validation_tokens,
        sequence_length=args.sequence_length,
        seed=args.seed,
        render_mode=args.render_mode,
    )
    print(
        json.dumps(
            {
                "event": "calibration_corpus_ready",
                "path": str(corpus.root),
                "train_tokens": corpus.token_count("train"),
                "validation_tokens": corpus.token_count("validation"),
            }
        )
    )
    return 0


def _trace_sources(values: list[str] | None):
    from mfq.calibration.dataset import TraceSource

    selected = values or [
        "nonthinking=calib_nonthinking_clean.jsonl",
        "thinking=calib_thinking_clean.jsonl",
    ]
    sources = []
    for value in selected:
        mode, separator, filename = value.partition("=")
        if not separator or mode not in {"thinking", "nonthinking"} or not filename:
            raise ValueError("--source-file must use thinking=FILE or nonthinking=FILE")
        sources.append(TraceSource(filename=filename, expected_mode=mode))
    return tuple(sources)


def _calibrate_trace_data(args: argparse.Namespace) -> int:
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("trace corpus generation requires transformers") from exc
    from mfq.calibration.dataset import build_hf_trace_corpus

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )
    corpus = build_hf_trace_corpus(
        tokenizer,
        args.output,
        repo_id=args.repo,
        revision=args.revision or None,
        sources=_trace_sources(args.source_file),
        expected_generator_model=args.expected_generator_model,
        cache_dir=args.cache_dir or None,
        train_tokens=args.train_tokens,
        validation_tokens=args.validation_tokens,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "event": "trace_calibration_corpus_ready",
                "path": str(corpus.root),
                "train_tokens": corpus.token_count("train"),
                "validation_tokens": corpus.token_count("validation"),
                "resolved_revision": corpus.manifest["sources"]["resolved_revision"],
                "token_sha256": corpus.manifest["token_sha256"],
            }
        )
    )
    return 0



def _calibrate_collect(args: argparse.Namespace) -> int:
    from mfq.calibration.dataset import load_corpus
    from mfq.calibration.statistics import collect_qwen35_statistics

    collect_qwen35_statistics(
        args.model,
        load_corpus(args.corpus),
        args.output,
        device=args.device,
        attention=args.attention,
        input_window=args.input_window,
        fisher_window=args.fisher_window,
        train_input_tokens=args.train_input_tokens,
        validation_input_tokens=args.validation_input_tokens,
        train_fisher_tokens=args.train_fisher_tokens,
        validation_fisher_tokens=args.validation_fisher_tokens,
        seed=args.seed,
        work_dir=args.work_dir or None,
        head_row_chunk=args.head_row_chunk,
        keep_hidden=args.keep_hidden,
    )
    return 0


def _calibrate_imatrix(args: argparse.Namespace) -> int:
    if getattr(args, "uniform", False):
        from mfq.calibration.uniform import uniform_imatrix

        result = uniform_imatrix(args.model, args.output)
        print(json.dumps({
            "event": "imatrix_saved",
            "output": str(result.path),
            "entries": len(result.entries),
            "objective": "uniform_calibration_free",
            "tokens": 0,
        }))
        return 0

    from mfq.calibration.chat import prepare_calibration_corpus
    from mfq.calibration.imatrix import collect_imatrix, imatrix_model_type

    try:
        imatrix_model_type(args.model)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    device = args.device or ("mps" if args.backend == "metal" else "cuda:0")
    device_type = device.split(":", 1)[0]
    expected_type = "mps" if args.backend == "metal" else "cuda"
    if device_type != expected_type:
        raise ValueError(
            f"--backend {args.backend} requires a {expected_type} --device, got {device!r}"
        )
    accumulation_dtype = args.accumulation_dtype
    if accumulation_dtype == "auto":
        accumulation_dtype = "float32" if args.backend == "metal" else "float64"
    with prepare_calibration_corpus(args.corpus, args.model,
        apply_chat_template=getattr(args, "render_mode", "auto") != "plain",
        window_length=args.window_length, train_tokens=args.train_tokens) as corpus:
        collect_imatrix(
            args.model,
            corpus,
            args.output,
            device=device,
            attention=args.attention,
            window_length=args.window_length,
            batch_size=args.batch_size,
            pad_to_multiple=getattr(args, "pad_to_multiple", 0) or None,
            train_tokens=args.train_tokens,
            seed=args.seed,
            work_dir=args.work_dir or None,
            keep_hidden=args.keep_hidden,
            accumulation_dtype=accumulation_dtype,
            objective=getattr(args, "objective", "naq"),
            layerwise=not getattr(args, "resident", False),
        )
    return 0


def _wt2_reference(args: argparse.Namespace) -> int:
    from mfq.evaluation.reference import generate_reference
    backend = ('metal' if sys.platform == 'darwin' else 'cuda') if args.backend == 'auto' else args.backend
    device = args.device or ('mps' if backend == 'metal' else 'cuda:0')
    if device.split(':', 1)[0] != ('mps' if backend == 'metal' else 'cuda'):
        raise ValueError('WT2 device must match the selected backend')
    generate_reference(args.model, args.dataset, args.output, context_size=args.context_size,
        chunks=args.chunks, device=device, layerwise=not args.resident, runtime=args.runtime or None)
    return 0


def _candidate_evaluations(args: argparse.Namespace):
    from mfq.calibration.artifact import ExpertPrecision
    from mfq.calibration.evaluator import NINT_EXPERT_PROFILES, evaluate_candidates
    from mfq.calibration.statistics import load_statistics

    selected = tuple(args.profile or ("NINT4", "NINT5", "NINT6", "NINT8"))
    artifacts = {}
    for value in args.precision_artifact:
        profile, separator, raw_path = value.partition("=")
        if not separator or not profile or not raw_path:
            raise ValueError("--precision-artifact must use PROFILE=PATH")
        if profile in artifacts:
            raise ValueError(f"duplicate precision artifact for {profile}")
        artifacts[profile] = str(Path(raw_path).resolve())
    unknown_artifacts = sorted(set(artifacts) - set(selected))
    if unknown_artifacts:
        raise ValueError(f"precision artifacts have no selected profile: {unknown_artifacts}")
    nvq_options = {
        "NVQ3J": (
            ("assignment_refine_steps", 2),
            ("banks", 2),
            ("codebook_train_rows", 2048),
            ("group_chunk", 1024),
            ("iterations", 4),
            ("learned_scale_lut", False),
            ("raw_multiplier", 8),
            ("search_steps", 19),
            ("seed", 20260716),
        ),
        "NVQ3J-L": (
            ("assignment_refine_steps", 2),
            ("banks", 2),
            ("codebook_train_rows", 2048),
            ("group_chunk", 1024),
            ("iterations", 4),
            ("learned_scale_lut", False),
            ("raw_multiplier", 8),
            ("search_steps", 19),
            ("seed", 20260716),
        ),
        "NVQ2J-XL": (
            ("assignment_refine_steps", 2),
            ("banks", 4),
            ("codebook_train_rows", 2048),
            ("group_chunk", 1024),
            ("iterations", 4),
            ("learned_scale_lut", True),
            ("raw_multiplier", 8),
            ("search_steps", 19),
            ("seed", 20260716),
        ),
    }
    profiles = {
        profile: (
            NINT_EXPERT_PROFILES[profile]
            if profile in NINT_EXPERT_PROFILES
            else ExpertPrecision(
                family=profile,
                artifact=artifacts.get(profile),
                options=nvq_options.get(profile, ()),
            )
        )
        for profile in selected
    }

    statistics = load_statistics(args.statistics)
    target_names = None
    raw_layers = getattr(args, "layers", "")
    if raw_layers:
        selected_layers: set[int] = set()
        for part in raw_layers.split(","):
            first, separator, last = part.strip().partition("-")
            if separator:
                selected_layers.update(range(int(first), int(last) + 1))
            else:
                selected_layers.add(int(first))
        from mfq.calibration.qwen35 import qwen35_linear_targets

        target_names = {
            target.name
            for target in qwen35_linear_targets(args.model)
            if int(target.name.split(".layers.", 1)[1].split(".", 1)[0]) in selected_layers
        }
        if not target_names:
            raise ValueError("--layers selected no Dense calibration targets")
    quantization_importance = None
    extra_identity = ""
    if args.codebook_imatrix:
        import hashlib

        import numpy as np

        from mfq.calibration.qwen35 import qwen35_linear_targets
        from mfq.quantize.imatrix import load_importance_matrix
        from mfq.tools.quantize_hf_to_mfq import _hf_to_gguf_name

        imatrix_path = Path(args.codebook_imatrix).resolve()
        digest = hashlib.sha256()
        with imatrix_path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        extra_identity = f"codebook-imatrix:{digest.hexdigest()}"
        imatrix = load_importance_matrix(imatrix_path)
        selected_entries = {
            name: entry
            for name, entry in statistics.entries.items()
            if target_names is None or name in target_names
        }
        gguf_names = {
            target.name: (
                _hf_to_gguf_name(target.source_name) or target.gguf_name
            )
            for target in qwen35_linear_targets(args.model)
        }
        quantization_importance = {}
        for name, entry in selected_entries.items():
            gguf_name = gguf_names.get(name)
            if gguf_name is None or gguf_name not in imatrix.entries:
                raise KeyError(f"missing codebook imatrix entry: {name}: {gguf_name}")
            values = np.ascontiguousarray(
                imatrix.entries[gguf_name].values[0],
                dtype=np.float32,
            )
            if values.shape != (entry.target.columns,):
                raise ValueError(
                    f"codebook imatrix width mismatch: {name}: "
                    f"{values.shape} != {(entry.target.columns,)}"
                )
            quantization_importance[name] = values
    evaluations = evaluate_candidates(
        args.model,
        statistics,
        profiles=profiles,
        cache_dir=args.candidate_cache,
        packed_cache_dir=args.packed_candidate_cache or None,
        backend=args.quant_backend,
        device=args.device,
        row_chunk=args.row_chunk,
        target_names=target_names,
        quantization_importance=quantization_importance,
        extra_identity=extra_identity,
    )
    return statistics, evaluations


def _calibrate_candidates(args: argparse.Namespace) -> int:
    _statistics, evaluations = _candidate_evaluations(args)
    print(
        json.dumps(
            {
                "event": "dense_candidates_ready",
                "tensors": len(evaluations),
                "candidates": sum(len(value) for value in evaluations.values()),
                "layers": args.layers,
            }
        )
    )
    return 0


def _calibrate_allocate(args: argparse.Namespace) -> int:
    from mfq.calibration.evaluator import allocate_scheme

    statistics, evaluations = _candidate_evaluations(args)
    scheme = allocate_scheme(
        evaluations,
        args.output,
        target_profile=args.target_profile,
        statistics=statistics,
        metadata={"model": str(Path(args.model).resolve())},
    )
    print(
        json.dumps(
            {
                "event": "calibration_scheme_ready",
                "path": str(scheme.path),
                "target_profile": scheme.target_profile,
                "storage_bits": scheme.storage_bits,
                "bpw": scheme.bpw,
            }
        )
    )
    return 0


def _calibrate_inint(args: argparse.Namespace) -> int:
    from mfq.calibration.inint import build_inint_selector

    _statistics, evaluations = _candidate_evaluations(args)
    selector = build_inint_selector(
        evaluations,
        args.output,
        target_profile=args.target_profile,
        exact_row_limit=args.exact_row_limit,
        boundary_rows=args.boundary_rows,
        metadata={"model": str(Path(args.model).resolve())},
    )
    print(
        json.dumps(
            {
                "event": "inint_selector_ready",
                "path": str(selector.path),
                "target_profile": selector.target_profile,
                "selected_rows": selector.selected_rows,
                "row_count": selector.row_count,
                **selector.metadata,
            }
        )
    )
    return 0



def _dense_profile(value: str) -> str:
    from mfq.calibration.evaluator import DENSE_PROFILE_FAMILIES

    if value not in DENSE_PROFILE_FAMILIES:
        supported = ", ".join(DENSE_PROFILE_FAMILIES)
        raise argparse.ArgumentTypeError(
            f"unsupported Dense profile {value!r}; choose one of: {supported}"
        )
    return value



def _add_candidate_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_target_profile: bool = True,
) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument(
        "--codebook-imatrix",
        default="",
        help="separate imatrix used only for NVQ codebook training",
    )
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--packed-candidate-cache", default="")
    parser.add_argument(
        "--profile",
        action="append",
        type=_dense_profile,
        metavar="PROFILE",
        default=None,
        help="repeat to select packed Dense candidate families",
    )
    parser.add_argument(
        "--precision-artifact",
        action="append",
        default=[],
        metavar="PROFILE=PATH",
    )
    if include_target_profile:
        parser.add_argument("--target-profile", required=True)
    parser.add_argument("--quant-backend", choices=("cuda", "metal", "cpu"), default="cuda")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--row-chunk", type=int, default=256)


def _add_calibration_parsers(sub: argparse._SubParsersAction) -> None:

    from mfq.commands.alphaq import add_parser as add_alphaq_parser
    from mfq.commands.natscale import add_parsers as add_natscale_parsers

    calibrate = sub.add_parser("calibrate", help="build calibration artifacts")
    stages = calibrate.add_subparsers(dest="calibration_stage", metavar="<stage>", required=True)
    add_alphaq_parser(stages)
    add_natscale_parsers(stages)


    data = stages.add_parser("data", help="tokenize eaddario calibration records")
    data.add_argument("--model", required=True, help="local Qwen3.5 tokenizer directory")
    data.add_argument("--output", required=True)
    data.add_argument("--repo", default="eaddario/imatrix-calibration")
    data.add_argument(
        "--source-size",
        choices=("micro", "tiny", "small", "medium", "large"),
        default="medium",
    )
    data.add_argument("--cache-dir", default="")
    data.add_argument("--proxy", default="")
    data.add_argument("--train-tokens", type=int, default=1_572_864)
    data.add_argument("--validation-tokens", type=int, default=262_144)
    data.add_argument("--sequence-length", type=int, default=2048)
    data.add_argument("--seed", type=int, default=20260718)
    data.add_argument("--render-mode", choices=("auto", "chat", "plain"), default="auto")
    data.set_defaults(_impl=_calibrate_data)

    trace_data = stages.add_parser("trace-data", help="tokenize model-generated HF JSONL traces")
    trace_data.add_argument("--model", required=True, help="local target tokenizer directory")
    trace_data.add_argument("--output", required=True)
    trace_data.add_argument("--repo", default="anm2211/Qwen3.5-9B-Calibration")
    trace_data.add_argument(
        "--revision",
        default="f39213d3aefe5eb8aaf40d4c321021782e64db35",
        help="immutable HF dataset revision",
    )
    trace_data.add_argument(
        "--source-file",
        action="append",
        default=None,
        metavar="MODE=FILE",
        help="repeat for thinking/nonthinking JSONL sources",
    )
    trace_data.add_argument("--expected-generator-model", default="qwen3.5-9b")
    trace_data.add_argument("--cache-dir", default="")
    trace_data.add_argument("--proxy", default="")
    trace_data.add_argument("--train-tokens", type=int, default=1_572_864)
    trace_data.add_argument("--validation-tokens", type=int, default=262_144)
    trace_data.add_argument("--sequence-length", type=int, default=2048)
    trace_data.add_argument("--seed", type=int, default=20260718)
    trace_data.set_defaults(_impl=_calibrate_trace_data)

    collect = stages.add_parser("collect", help="collect activation and Fisher statistics")
    collect.add_argument("--model", required=True)
    collect.add_argument("--corpus", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--device", default="cuda:0")
    collect.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    collect.add_argument("--input-window", type=int, default=2048)
    collect.add_argument("--fisher-window", type=int, default=128)
    collect.add_argument("--train-input-tokens", type=int, default=1_572_864)
    collect.add_argument("--validation-input-tokens", type=int, default=262_144)
    collect.add_argument("--train-fisher-tokens", type=int, default=65_536)
    collect.add_argument("--validation-fisher-tokens", type=int, default=16_384)
    collect.add_argument("--seed", type=int, default=20260718)
    collect.add_argument("--work-dir", default="")
    collect.add_argument(
        "--head-row-chunk",
        type=int,
        default=0,
        help="lm_head rows per chunk; 0 uses exact full-head cross-entropy",
    )
    collect.add_argument("--keep-hidden", action="store_true")
    collect.set_defaults(_impl=_calibrate_collect)

    imatrix = stages.add_parser(
        "imatrix",
        help="create reusable activation or calibration-free uniform importance",
    )
    imatrix.add_argument("--model", required=True, help="local full-precision HF model")
    imatrix_source = imatrix.add_mutually_exclusive_group(required=True)
    imatrix_source.add_argument("--corpus", help=".txt/.json/.jsonl file or prepared MFQ calibration corpus")
    imatrix_source.add_argument(
        "--uniform", action="store_true",
        help="calibration-free all-ones importance from config and tensor headers; includes vision and MTP",
    )
    imatrix.add_argument("--render-mode", choices=("auto", "plain"), default="auto",
        help="auto adds the model chat template only to unformatted records; plain disables automatic wrapping")
    imatrix.add_argument("--output", required=True, help="native MFQ imatrix artifact")
    imatrix.add_argument("--backend", choices=("cuda", "metal"), default="cuda")
    imatrix.add_argument(
        "--device",
        default="",
        help="override accelerator device; defaults to cuda:0 or mps",
    )
    imatrix.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    imatrix.add_argument(
        "--objective",
        choices=("naq", "linear"),
        default="naq",
        help="NAQ-imatrix records input-channel and output-neuron importance; linear records only input second moments",
    )
    imatrix.add_argument("--window-length", type=int, default=16_384)
    imatrix.add_argument("--batch-size", type=int, default=1)
    imatrix.add_argument(
        "--pad-to-multiple",
        type=int,
        default=0,
        help="right-pad nearby-length traces for batching; 0 keeps exact-length batches",
    )
    imatrix.add_argument("--train-tokens", type=int, default=1_572_864)
    imatrix.add_argument("--seed", type=int, default=20260810)
    imatrix.add_argument("--work-dir", default="")
    imatrix.add_argument(
        "--accumulation-dtype",
        choices=("auto", "float32", "float64"),
        default="auto",
        help="auto uses FP64 on CUDA and FP32 on Metal",
    )
    imatrix.add_argument("--keep-hidden", action="store_true")
    imatrix.add_argument("--resident", action="store_true", help="keep source layers resident when current free memory permits; otherwise fall back to layerwise")
    imatrix.set_defaults(_impl=_calibrate_imatrix)

    allocate = stages.add_parser("allocate", help="score candidates and allocate tensor precision")
    _add_candidate_arguments(allocate)
    allocate.add_argument("--output", required=True)
    allocate.set_defaults(_impl=_calibrate_allocate)

    candidates = stages.add_parser(
        "candidates", help="materialize packed Dense candidates without allocating a scheme"
    )
    _add_candidate_arguments(candidates, include_target_profile=False)
    candidates.add_argument("--layers", default="", help="comma-separated layers or ranges")
    candidates.set_defaults(_impl=_calibrate_candidates)

    inint = stages.add_parser("inint", help="select per-neuron NINT4/NINT8 rows")
    _add_candidate_arguments(inint)
    inint.add_argument("--output", required=True)
    inint.add_argument("--exact-row-limit", type=int, default=100_000)
    inint.add_argument("--boundary-rows", type=int, default=32_768)
    inint.set_defaults(_impl=_calibrate_inint)




def _build_parser() -> argparse.ArgumentParser:
    from mfq.commands.build import add_parser as add_build_parser
    from mfq.commands.optimize import add_parser as add_optimize_parser
    from mfq.commands.quantize import add_parser as add_quantize_parser
    from mfq.commands.serve import add_parser as add_serve_parser
    from mfq.commands.solve_ew import add_parser as add_solve_ew_parser

    parser = argparse.ArgumentParser(prog="mfq", description="Mixed Format Quantization toolchain.")
    parser.add_argument("-V", "--version", action="version", version=f"mfq {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    add_build_parser(sub)
    add_optimize_parser(sub)
    add_serve_parser(sub)
    add_quantize_parser(sub)
    add_solve_ew_parser(sub)
    _add_calibration_parsers(sub)
    reference = sub.add_parser('wt2-reference', help='generate official WT2 reference logits from original HF weights')
    reference.add_argument('--model', required=True)
    reference.add_argument('--dataset', required=True)
    reference.add_argument('--output', required=True)
    reference.add_argument('--context-size', type=int, default=512)
    reference.add_argument('--chunks', type=int, default=8)
    reference.add_argument('--backend', choices=('auto', 'metal', 'cuda'), default='auto')
    reference.add_argument('--device', default='')
    reference.add_argument('--resident', action='store_true')
    reference.add_argument('--runtime', default='', help='native CUDA runtime executable')
    reference.set_defaults(_impl=_wt2_reference)
    sub.add_parser(
        "voice-runtime-check",
        help="verify optional MiniCPM-o voice output dependencies",
    ).set_defaults(_impl=_voice_runtime_check)
    flash_next_worker = sub.add_parser("_flash-next-worker", help=argparse.SUPPRESS)
    flash_next_worker.add_argument("--model", type=Path, required=True)
    flash_next_worker.add_argument("--host", default="127.0.0.1")
    flash_next_worker.add_argument("--port", type=int, required=True)
    flash_next_worker.add_argument("--model-name", required=True)
    flash_next_worker.add_argument("--ctx-size", dest="context_size", type=int, default=0)
    flash_next_worker.add_argument("--prefill-chunk-size", type=int, default=2_048)
    flash_next_worker.set_defaults(_impl=_flash_next_worker)
    sub.add_parser("_mlx-runtime-check", help=argparse.SUPPRESS).set_defaults(
        _impl=_mlx_runtime_check
    )
    hub_download = sub.add_parser("_hub-download", help=argparse.SUPPRESS)
    hub_download.add_argument(
        "--provider", required=True, choices=("huggingface", "modelscope")
    )
    hub_download.add_argument("--repo-id", required=True)
    hub_download.add_argument("--repo-type", default="model")
    hub_download.add_argument("--revision", required=True)
    hub_download.add_argument("--local-dir", type=Path, required=True)
    hub_download.add_argument("--max-workers", type=int, default=8)
    hub_download.add_argument("--include", action="append", default=[])
    hub_download.add_argument("--exclude", action="append", default=[])
    hub_download.set_defaults(_impl=_hub_download)
    sub.add_parser("inspect", help="inspect an MFQ file").set_defaults(_impl=_not_implemented)
    return parser


def main(argv: list[str] | None = None) -> int:
    if bool(getattr(sys, "frozen", False)):
        import multiprocessing

        multiprocessing.freeze_support()
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == '_quantization-worker':
        from mfq.quantize.workbench import main as quantization_main
        quantization_main(arguments[1:])
        return 0
    if arguments == ["_official-scoring-probe"]:
        print(4)
        return 0
    if arguments == ["_official-scoring-worker"]:
        from mfq.server.services.official_scoring_worker import main as score_main

        score_main()
        return 0
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args._impl(args))


if __name__ == "__main__":
    raise SystemExit(main())
