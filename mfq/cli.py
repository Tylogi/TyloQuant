"""MFQ command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mfq._version import __version__


def _not_implemented(args: argparse.Namespace) -> int:
    print(f"command {args.command!r} is not implemented", file=sys.stderr)
    return 2


def _tpq_inspect(args: argparse.Namespace) -> int:
    from mfq.runtime.tpq import open_tpq_artifact

    artifact = open_tpq_artifact(args.model)
    print(json.dumps(artifact.summary(), ensure_ascii=False, indent=2))
    return 0


def _tpq_import(args: argparse.Namespace) -> int:
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


def _tpq_train_v4f(args: argparse.Namespace) -> int:
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


def _tpq_quantize_v4f(args: argparse.Namespace) -> int:
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


def _tpq_prepare(args: argparse.Namespace) -> int:
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


def _tpq_run(args: argparse.Namespace) -> int:
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
) -> None:
    tpq = sub.add_parser(
        command,
        help="inspect, convert, quantize, or run a TPQ model",
    )
    stages = tpq.add_subparsers(
        dest=f"{command}_stage",
        metavar="<stage>",
        required=True,
    )

    inspect = stages.add_parser("inspect", help="validate and inspect a TPQ model")
    inspect.add_argument("model")
    inspect.set_defaults(_impl=_tpq_inspect)

    import_model = stages.add_parser(
        "import",
        help="stream a TPQ directory into one native MFQ file",
    )
    import_model.add_argument("--input", required=True)
    import_model.add_argument("--output", required=True)
    import_model.add_argument("--row-chunk", type=int, default=4096)
    import_model.add_argument("--workers", type=int, default=8)
    import_model.add_argument("--overwrite", action="store_true")
    import_model.set_defaults(_impl=_tpq_import)

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
    prepare.set_defaults(_impl=_tpq_prepare)

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
    train_v4f.set_defaults(_impl=_tpq_train_v4f)

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
    quantize_v4f.set_defaults(_impl=_tpq_quantize_v4f)

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
    run.set_defaults(_impl=_tpq_run)


def _add_tpq_parsers(sub: argparse._SubParsersAction) -> None:
    _add_tpq_parser(sub, "tpq")


def _build_parser() -> argparse.ArgumentParser:
    from mfq.commands.quantize import add_parser as add_quantize_parser
    from mfq.commands.solve_ew import add_parser as add_solve_ew_parser

    parser = argparse.ArgumentParser(prog="mfq", description="Mixed Format Quantization toolchain.")
    parser.add_argument("-V", "--version", action="version", version=f"mfq {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    add_quantize_parser(sub)
    add_solve_ew_parser(sub)
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
