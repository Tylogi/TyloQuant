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


def _candidate_evaluations(args: argparse.Namespace):
    from mfq.calibration.evaluator import evaluate_candidates
    from mfq.calibration.statistics import load_statistics

    statistics = load_statistics(args.statistics)
    evaluations = evaluate_candidates(
        args.model,
        statistics,
        cache_dir=args.candidate_cache,
        packed_cache_dir=args.packed_candidate_cache or None,
        backend=args.quant_backend,
        device=args.device,
        row_chunk=args.row_chunk,
    )
    return statistics, evaluations


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


def _calibrate_validate(args: argparse.Namespace) -> int:
    from mfq.calibration.artifact import load_scheme
    from mfq.calibration.collector import validate_layerwise
    from mfq.calibration.dataset import load_corpus
    from mfq.calibration.layerwise_qwen35 import Qwen35LayerwiseBackend

    backend = Qwen35LayerwiseBackend(
        args.model,
        load_scheme(args.scheme),
        device=args.device,
        quant_backend=args.quant_backend,
    )
    traces = validate_layerwise(
        backend,
        load_corpus(args.corpus),
        args.output,
        work_dir=args.work_dir,
        split=args.split,
        window_length=args.window_length,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        seed=args.seed,
        keep_hidden=args.keep_hidden,
    )
    print(
        json.dumps(
            {
                "event": "layerwise_validation_ready",
                "path": str(Path(args.output).resolve()),
                "layers": len(traces),
                "final_relative_rmse_percent": traces[-1].relative_rmse_percent,
                "final_cosine_similarity": traces[-1].cosine_similarity,
            }
        )
    )
    return 0


def _calibrate_refine(args: argparse.Namespace) -> int:
    from mfq.calibration.artifact import load_scheme
    from mfq.calibration.dataset import load_corpus
    from mfq.calibration.evaluator import (
        build_compensated_layer_strategies,
        build_global_layer_strategies,
        build_layer_strategies,
    )
    from mfq.calibration.layerwise_qwen35 import Qwen35LayerwiseBackend
    from mfq.calibration.refinement import refine_layerwise, refine_layerwise_global
    from mfq.calibration.soft_refinement import refine_soft_rate_distortion

    base_scheme = load_scheme(args.scheme)
    precision_groups = None
    if args.allocation == "soft-global":
        from mfq.calibration.evaluator import load_scheme_candidate_evaluations
        from mfq.calibration.rate_distortion import build_precision_groups_from_scheme

        evaluations, candidate_audit = load_scheme_candidate_evaluations(
            base_scheme,
            args.candidate_cache,
        )
        print(json.dumps({"event": "scheme_candidate_cache_ready", **candidate_audit}))
        precision_groups = build_precision_groups_from_scheme(base_scheme)
    else:
        _statistics, evaluations = _candidate_evaluations(args)
    if args.allocation == "global":
        strategies = build_global_layer_strategies(evaluations, base_scheme)
    elif args.allocation == "compensated":
        strategies = build_compensated_layer_strategies(evaluations, base_scheme)
    elif args.allocation == "greedy":
        strategies = build_layer_strategies(
            evaluations,
            base_scheme,
            top_k=args.top_k,
        )
    else:
        strategies = None
    backend = Qwen35LayerwiseBackend(
        args.model,
        base_scheme,
        device=args.device,
        quant_backend=args.quant_backend,
        candidate_cache_dir=(
            Path(args.packed_candidate_cache).resolve()
            if args.allocation == "soft-global" and args.packed_candidate_cache
            else (
                Path(args.work_dir) / "packed-candidates"
                if args.allocation == "soft-global"
                else None
            )
        ),
    )
    refine_batch_size = args.batch_size or (8 if args.allocation == "soft-global" else 1)
    refine_tokens = args.max_tokens
    if refine_tokens is None:
        refine_tokens = 1_572_864 if args.allocation == "soft-global" else 8_192
    if args.allocation == "soft-global":
        result = refine_soft_rate_distortion(
            backend,
            load_corpus(args.corpus),
            base_scheme,
            evaluations,
            args.output,
            args.report,
            work_dir=args.work_dir,
            window_length=args.window_length,
            batch_size=refine_batch_size,
            train_tokens=refine_tokens,
            validation_tokens=args.validation_tokens,
            validation_corpus=(
                load_corpus(args.validation_corpus) if args.validation_corpus else None
            ),
            shard_tokens=args.soft_shard_tokens,
            epochs=args.soft_epochs,
            hard_train_tokens=args.hard_train_tokens,
            seed=args.seed,
            steps=args.soft_steps,
            learning_rate=args.soft_learning_rate,
            dual_learning_rate=args.dual_learning_rate,
            initial_dual=args.initial_dual,
            temperature_start=args.temperature_start,
            temperature_end=args.temperature_end,
            head_row_chunk=args.head_row_chunk,
            discrete_iterations=args.discrete_iterations,
            discrete_single_candidates=args.discrete_single_candidates,
            discrete_pair_candidates=args.discrete_pair_candidates,
            validation_tolerance=args.validation_tolerance,
            resume=args.resume,
            keep_hidden=args.keep_hidden,
            precision_groups=precision_groups,
        )
        scheme = result.scheme
        completed_iterations = len(result.steps)
    elif args.allocation == "global":
        assert strategies is not None
        scheme, rounds = refine_layerwise_global(
            backend,
            load_corpus(args.corpus),
            base_scheme,
            evaluations,
            strategies,
            args.output,
            args.report,
            work_dir=args.work_dir,
            window_length=args.window_length,
            batch_size=refine_batch_size,
            max_tokens=refine_tokens,
            seed=args.seed,
            max_iterations=args.iterations,
            max_changed_layers=args.max_changed_layers,
            keep_hidden=args.keep_hidden,
        )
        completed_iterations = len(rounds)
    else:
        assert strategies is not None
        scheme, refinements = refine_layerwise(
            backend,
            load_corpus(args.corpus),
            base_scheme,
            evaluations,
            strategies,
            args.output,
            args.report,
            work_dir=args.work_dir,
            window_length=args.window_length,
            batch_size=refine_batch_size,
            max_tokens=refine_tokens,
            seed=args.seed,
            cumulative_budget=args.allocation == "compensated",
            keep_hidden=args.keep_hidden,
        )
        completed_iterations = 1
    print(
        json.dumps(
            {
                "event": "layerwise_refinement_ready",
                "path": str(scheme.path),
                "report": str(Path(args.report).resolve()),
                "allocation": args.allocation,
                "layers": backend.num_layers,
                "iterations": completed_iterations,
                "storage_bits": scheme.storage_bits,
                "bpw": scheme.bpw,
            }
        )
    )
    return 0


def _cccp_inspect(args: argparse.Namespace) -> int:
    from mfq.runtime.tpq import open_tpq_artifact

    artifact = open_tpq_artifact(args.model)
    print(json.dumps(artifact.summary(), ensure_ascii=False, indent=2))
    return 0


def _cccp_import(args: argparse.Namespace) -> int:
    from mfq.tools.import_tpq_to_mfq import convert

    output = convert(
        args.input,
        args.output,
        row_chunk=args.row_chunk,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": output.stat().st_size,
                "gb": output.stat().st_size / 1e9,
            }
        )
    )
    return 0


def _cccp_train_v4f(args: argparse.Namespace) -> int:
    from mfq.tools.train_tpq_v4f_codebooks import _layers, train

    events = train(
        input_root=args.input,
        scheme_path=args.scheme,
        device=args.device,
        points_per_expert=args.points_per_expert,
        vv_points_per_expert=args.vv_points_per_expert,
        max_experts=args.max_experts,
        layers=_layers(args.layers),
        overwrite=args.overwrite,
    )
    print(json.dumps({"trained_cohorts": len(events)}))
    return 0


def _cccp_quantize_v4f(args: argparse.Namespace) -> int:
    from mfq.tools.quantize_tpq_v4f_to_mfq import convert

    if not args.work_dir:
        args.work_dir = str(Path(args.scheme).resolve().parent)
    convert(
        input_root=args.input,
        scheme_path=args.scheme,
        output_path=args.output,
        work_dir=args.work_dir,
        temp_dir=args.temp_dir or None,
        device=args.device,
        row_chunk=args.row_chunk,
        overwrite=args.overwrite,
    )
    return 0


def _cccp_prepare(args: argparse.Namespace) -> int:
    from mfq.tools.prepare_tpq_scheme import prepare

    scheme = prepare(
        profile_path=args.profile,
        output_path=args.output,
        field=args.field,
        rows_gate_up=args.rows_gate_up,
        columns_gate_up=args.columns_gate_up,
        rows_down=args.rows_down,
        columns_down=args.columns_down,
        v_coverage=args.v_coverage,
        w_coverage=args.w_coverage,
        vv_share=args.vv_share,
    )
    print(
        json.dumps(
            {
                "output": str(scheme.path),
                "expert_tensors": len(scheme.expert_selections),
                "storage_bits": scheme.storage_bits,
                "bpw": scheme.bpw,
            }
        )
    )
    return 0


def _cccp_run(args: argparse.Namespace) -> int:
    from mfq.runtime.tpq import (
        configure_tpq_memory,
        open_tpq_artifact,
        run_tpq_chat,
    )

    artifact = open_tpq_artifact(args.model)
    memory = configure_tpq_memory(artifact)
    cache_gb = args.cache_gb
    if cache_gb is None and memory["recommended_cache_gib"] is not None:
        cache_gb = min(32.0, float(memory["recommended_cache_gib"]))
        print(
            "[mfq] TPQ cgroup-aware host cache: "
            f"{cache_gb:.1f} GiB; full_resident="
            f"{os.environ.get('TPQ_FULL_RESIDENT', 'auto')}",
            flush=True,
        )
    runtime_args = [
        "--model",
        str(artifact.path if hasattr(artifact, "path") else artifact.root),
        "--device",
        args.device,
        "--max-new",
        str(args.max_new),
        "--temp",
        str(args.temp),
        "--top-p",
        str(args.top_p),
        "--rep-penalty",
        str(args.rep_penalty),
        "--no-repeat-ngram",
        str(args.no_repeat_ngram),
        "--spec",
        str(args.spec),
    ]
    optional_values = (
        ("--cache-gb", cache_gb),
        ("--vram-gb", args.vram_gb),
        ("--max-ctx", args.max_ctx),
        ("--tp", args.tp),
        ("--prompt", args.prompt),
        ("--tokenizer-root", args.tokenizer_root),
    )
    for option, value in optional_values:
        if value is not None:
            runtime_args.extend((option, str(value)))
    if args.no_max_new:
        runtime_args.append("--no-max-new")
    if args.think:
        runtime_args.append("--think")
    run_tpq_chat(runtime_args, tpq_root=args.tpq_root)
    return 0


def _add_tpq_parser(
    sub: argparse._SubParsersAction,
    command: str,
    *,
    legacy: bool = False,
) -> None:
    description = (
        "legacy alias for the TPQ commands"
        if legacy
        else "inspect, convert, quantize, or run a TPQ model"
    )
    tpq = sub.add_parser(command, help=description)
    stages = tpq.add_subparsers(
        dest=f"{command}_stage",
        metavar="<stage>",
        required=True,
    )

    inspect = stages.add_parser("inspect", help="validate and inspect a TPQ model")
    inspect.add_argument("model")
    inspect.set_defaults(_impl=_cccp_inspect)

    import_model = stages.add_parser(
        "import",
        help="stream a TPQ or legacy cccp-1 directory into one native MFQ file",
    )
    import_model.add_argument("--input", required=True)
    import_model.add_argument("--output", required=True)
    import_model.add_argument("--row-chunk", type=int, default=4096)
    import_model.add_argument("--workers", type=int, default=8)
    import_model.add_argument("--overwrite", action="store_true")
    import_model.set_defaults(_impl=_cccp_import)

    prepare = stages.add_parser(
        "prepare",
        help="prepare TPQ tiers from scores or a fixed tiers_per_layer map",
    )
    prepare.add_argument("--profile", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--field", default="counts")
    prepare.add_argument("--rows-gate-up", type=int, default=4096)
    prepare.add_argument("--columns-gate-up", type=int, default=4096)
    prepare.add_argument("--rows-down", type=int, default=4096)
    prepare.add_argument("--columns-down", type=int, default=2048)
    prepare.add_argument("--v-coverage", type=float, default=0.965)
    prepare.add_argument("--w-coverage", type=float, default=0.997)
    prepare.add_argument("--vv-share", type=float, default=0.25)
    prepare.set_defaults(_impl=_cccp_prepare)

    train_v4f = stages.add_parser(
        "train-v4f",
        help="train original weight-only TPQ codebooks for a V4F scheme",
    )
    train_v4f.add_argument("--input", required=True)
    train_v4f.add_argument("--scheme", required=True)
    train_v4f.add_argument("--device", default="cuda")
    train_v4f.add_argument("--points-per-expert", type=int, default=50_000)
    train_v4f.add_argument(
        "--vv-points-per-expert",
        type=int,
        default=300_000,
    )
    train_v4f.add_argument("--max-experts", type=int, default=32)
    train_v4f.add_argument("--layers", default="")
    train_v4f.add_argument("--overwrite", action="store_true")
    train_v4f.set_defaults(_impl=_cccp_train_v4f)

    quantize_v4f = stages.add_parser(
        "quantize-v4f",
        help="run original TPQ V4F quantization into one native MFQ file",
    )
    quantize_v4f.add_argument("--input", required=True)
    quantize_v4f.add_argument("--scheme", required=True)
    quantize_v4f.add_argument("--output", required=True)
    quantize_v4f.add_argument("--work-dir", default="")
    quantize_v4f.add_argument("--temp-dir", default="")
    quantize_v4f.add_argument("--device", default="cuda")
    quantize_v4f.add_argument("--row-chunk", type=int, default=512)
    quantize_v4f.add_argument("--overwrite", action="store_true")
    quantize_v4f.set_defaults(_impl=_cccp_quantize_v4f)

    run = stages.add_parser("run", help="run a TPQ model through fused TyloQuant kernels")
    run.add_argument("model")
    run.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run.add_argument("--cache-gb", type=float)
    run.add_argument("--vram-gb", type=float)
    run.add_argument("--max-ctx", type=int)
    run.add_argument("--tp", type=int)
    run.add_argument("--max-new", type=int, default=128)
    run.add_argument("--no-max-new", action="store_true")
    run.add_argument("--temp", type=float, default=0.0)
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument("--rep-penalty", type=float, default=1.0)
    run.add_argument("--no-repeat-ngram", type=int, default=0)
    run.add_argument("--spec", type=int, default=0)
    run.add_argument("--think", action="store_true")
    run.add_argument("--prompt")
    run.add_argument(
        "--tokenizer-root",
        help="tokenizer/config directory used with a native TPQ MFQ file",
    )
    run.add_argument("--tpq-root")
    run.set_defaults(_impl=_cccp_run)


def _add_tpq_parsers(sub: argparse._SubParsersAction) -> None:
    _add_tpq_parser(sub, "tpq")
    _add_tpq_parser(sub, "cccp", legacy=True)


def _add_cccp_parsers(sub: argparse._SubParsersAction) -> None:
    """Compatibility wrapper for callers that build only the old CLI group."""

    _add_tpq_parser(sub, "cccp", legacy=True)


def _add_candidate_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_target_profile: bool = True,
) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--packed-candidate-cache", default="")
    if include_target_profile:
        parser.add_argument("--target-profile", choices=("NINT5", "NINT6"), required=True)
    parser.add_argument("--quant-backend", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--row-chunk", type=int, default=256)


def _add_calibration_parsers(sub: argparse._SubParsersAction) -> None:
    calibrate = sub.add_parser("calibrate", help="build calibration artifacts")
    stages = calibrate.add_subparsers(dest="calibration_stage", metavar="<stage>", required=True)

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
    data.add_argument("--render-mode", choices=("chat", "plain"), default="plain")
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

    allocate = stages.add_parser("allocate", help="score candidates and allocate tensor precision")
    _add_candidate_arguments(allocate)
    allocate.add_argument("--output", required=True)
    allocate.set_defaults(_impl=_calibrate_allocate)

    inint = stages.add_parser("inint", help="select per-neuron NINT4/NINT8 rows")
    _add_candidate_arguments(inint)
    inint.add_argument("--output", required=True)
    inint.add_argument("--exact-row-limit", type=int, default=100_000)
    inint.add_argument("--boundary-rows", type=int, default=32_768)
    inint.set_defaults(_impl=_calibrate_inint)

    refine = stages.add_parser(
        "refine", help="select layer strategies on the cumulative quantized path"
    )
    _add_candidate_arguments(refine, include_target_profile=False)
    refine.add_argument("--corpus", required=True)
    refine.add_argument("--validation-corpus", default="")
    refine.add_argument("--scheme", required=True)
    refine.add_argument("--output", required=True)
    refine.add_argument("--report", required=True)
    refine.add_argument("--work-dir", required=True)
    refine.add_argument(
        "--allocation",
        choices=("soft-global", "compensated", "greedy", "global"),
        default="soft-global",
    )
    refine.add_argument("--iterations", type=int, default=3)
    refine.add_argument("--max-changed-layers", type=int, default=0)
    refine.add_argument("--top-k", type=int, default=8)
    refine.add_argument("--window-length", type=int, default=256)
    refine.add_argument("--batch-size", type=int, default=None)
    refine.add_argument("--max-tokens", type=int, default=None)
    refine.add_argument("--validation-tokens", type=int, default=262_144)
    refine.add_argument("--hard-train-tokens", type=int, default=262_144)
    refine.add_argument("--soft-shard-tokens", type=int, default=32_768)
    refine.add_argument("--soft-epochs", type=int, default=1)
    refine.add_argument("--soft-steps", type=int, default=20)
    refine.add_argument("--soft-learning-rate", type=float, default=0.08)
    refine.add_argument("--dual-learning-rate", type=float, default=0.1)
    refine.add_argument("--initial-dual", type=float, default=0.0)
    refine.add_argument("--temperature-start", type=float, default=1.5)
    refine.add_argument("--temperature-end", type=float, default=0.25)
    refine.add_argument("--head-row-chunk", type=int, default=2_048)
    refine.add_argument("--discrete-iterations", type=int, default=0)
    refine.add_argument("--discrete-single-candidates", type=int, default=8)
    refine.add_argument("--discrete-pair-candidates", type=int, default=16)
    refine.add_argument("--validation-tolerance", type=float, default=0.0)
    refine.add_argument("--resume", action="store_true")
    refine.add_argument("--seed", type=int, default=20260718)
    refine.add_argument("--keep-hidden", action="store_true")
    refine.set_defaults(_impl=_calibrate_refine)

    validate = stages.add_parser("validate", help="run cumulative Qwen3.5 layerwise replay")
    validate.add_argument("--model", required=True)
    validate.add_argument("--corpus", required=True)
    validate.add_argument("--scheme", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--work-dir", required=True)
    validate.add_argument("--split", choices=("train", "validation"), default="validation")
    validate.add_argument("--window-length", type=int, default=2048)
    validate.add_argument("--batch-size", type=int, default=1)
    validate.add_argument("--max-tokens", type=int, default=16_384)
    validate.add_argument("--seed", type=int, default=20260718)
    validate.add_argument("--device", default="cuda:0")
    validate.add_argument("--quant-backend", choices=("cuda", "cpu"), default="cuda")
    validate.add_argument("--keep-hidden", action="store_true")
    validate.set_defaults(_impl=_calibrate_validate)


def _build_parser() -> argparse.ArgumentParser:
    from mfq.commands.quantize import add_parser as add_quantize_parser

    parser = argparse.ArgumentParser(prog="mfq", description="Mixed Format Quantization toolchain.")
    parser.add_argument("-V", "--version", action="version", version=f"mfq {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    add_quantize_parser(sub)
    _add_calibration_parsers(sub)
    _add_tpq_parsers(sub)
    sub.add_parser("inspect", help="inspect an MFQ file").set_defaults(_impl=_not_implemented)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args._impl(args))


if __name__ == "__main__":
    raise SystemExit(main())
