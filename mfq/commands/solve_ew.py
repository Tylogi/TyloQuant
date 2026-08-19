"""Public command for generic expert-wise joint budget allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mfq.calibration.artifact import ExpertPrecision
    from mfq.calibration.ew_solver import EwCandidateTable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _rebase_precision(
    precision: ExpertPrecision,
    *,
    candidate_table: EwCandidateTable,
    output_root: Path,
) -> ExpertPrecision:
    if precision.artifact is None or candidate_table.artifact_root is None:
        return precision
    raw = Path(precision.artifact)
    source = raw if raw.is_absolute() else candidate_table.artifact_root / raw
    relative = os.path.relpath(source.resolve(), output_root)
    return replace(precision, artifact=Path(relative).as_posix())


def _rebase_scheme_artifacts(scheme: Any, candidate_table: EwCandidateTable, output: Path) -> Any:
    expert_selections = {}
    for name, tensor in scheme.expert_selections.items():
        experts = tuple(
            replace(
                expert,
                precision=_rebase_precision(
                    expert.descriptor,
                    candidate_table=candidate_table,
                    output_root=output.parent,
                ),
            )
            for expert in tensor.selections
        )
        expert_selections[name] = replace(tensor, selections=experts)
    return replace(scheme, path=output, expert_selections=expert_selections)


def run(args: argparse.Namespace) -> int:
    from mfq.calibration.artifact import save_scheme
    from mfq.calibration.ew_solver import (
        load_budget,
        load_candidate_table,
        load_importance_table,
        solve_ew_budget,
    )

    importance_path = Path(args.importance).resolve()
    candidate_path = Path(args.candidates).resolve()
    budget_path = Path(args.budget).resolve()
    output_scheme = Path(args.output_scheme).resolve()
    output_report = Path(args.report).resolve()
    if output_scheme == output_report:
        raise ValueError("output scheme and report paths must differ")
    for path in (output_scheme, output_report):
        if path.exists():
            raise FileExistsError(f"output already exists: {path}")

    importance = load_importance_table(importance_path)
    candidates = load_candidate_table(candidate_path)
    budget = load_budget(budget_path)
    result = solve_ew_budget(importance, candidates, budget)
    scheme = _rebase_scheme_artifacts(result.scheme, candidates, output_scheme)
    save_scheme(output_scheme, scheme)

    report = {
        **dict(result.report),
        "importance_path": str(importance_path),
        "importance_sha256": _sha256(importance_path),
        "candidate_path": str(candidate_path),
        "candidate_sha256": _sha256(candidate_path),
        "budget_path": str(budget_path),
        "budget_sha256": _sha256(budget_path),
        "scheme_path": str(output_scheme),
        "scheme_sha256": _sha256(output_scheme),
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_report.with_suffix(output_report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_report)
    print(
        json.dumps(
            {
                "status": "ok",
                "scheme": str(output_scheme),
                "report": str(output_report),
                "model_bpw": report["model_bpw"],
                "routed_bpw": report["routed_bpw"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "solve-ew",
        help="solve a joint expert-wise precision budget",
        description=(
            "Allocate expert precision from a score or rank importance table, "
            "an exact rate-distortion candidate table, and joint model/projection/"
            "layer/shape constraints."
        ),
    )
    parser.add_argument("--importance", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--budget", required=True)
    parser.add_argument("--output-scheme", required=True)
    parser.add_argument("--report", required=True)
    parser.set_defaults(_impl=run)


__all__ = ["add_parser", "run"]
