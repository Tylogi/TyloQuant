"""Allocate routed-expert precision under independently enforced layer budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from mfq.calibration.artifact import CalibrationScheme, save_scheme
from mfq.calibration.reap_expertwise import (
    ExpertProfileEvaluation,
    allocate_independent_expert_profiles,
    evaluation_from_document,
    load_reap_expert_table,
)
from mfq.tools.quantize_hf_to_mfq import _hf_to_gguf_name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_candidates(path: Path) -> list[ExpertProfileEvaluation]:
    values: list[ExpertProfileEvaluation] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                values.append(evaluation_from_document(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid expert candidate row {line_number} in {path}"
                ) from exc
    if not values:
        raise ValueError("candidate table is empty")
    return values


def _load_budget_document(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format") != "mfq.ud-layer-budgets.v1":
        raise ValueError("unsupported layer-budget format")
    expected_layers = int(document["expected_layers"])
    expected_experts = int(document["expected_experts"])
    expected_top_k = int(document["expected_top_k"])
    if expected_layers <= 0 or expected_experts <= 0 or expected_top_k <= 0:
        raise ValueError("layer-budget dimensions must be positive")
    layers = document.get("layers")
    if not isinstance(layers, dict) or set(layers) != {
        str(layer) for layer in range(expected_layers)
    }:
        raise ValueError("layer-budget document does not cover every layer")
    for layer in range(expected_layers):
        bits = int(layers[str(layer)]["target_storage_bits"])
        if bits <= 0:
            raise ValueError(f"layer {layer} storage budget must be positive")
    return document


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def allocate_reap_layer_budget(
    *,
    reap_path: str | Path,
    candidate_table_path: str | Path,
    layer_budgets_path: str | Path,
    output_scheme_path: str | Path,
    output_report_path: str | Path,
    target_label: str,
    tensor_namespace: str = "gguf",
) -> dict[str, Any]:
    reap = Path(reap_path).resolve()
    candidates = Path(candidate_table_path).resolve()
    budgets = Path(layer_budgets_path).resolve()
    output_scheme = Path(output_scheme_path).resolve()
    output_report = Path(output_report_path).resolve()
    for source in (reap, candidates, budgets):
        if not source.is_file():
            raise FileNotFoundError(source)
    for output in (output_scheme, output_report):
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
    if not target_label:
        raise ValueError("target label cannot be empty")
    if tensor_namespace not in {"gguf", "huggingface"}:
        raise ValueError("tensor_namespace must be gguf or huggingface")

    budget_document = _load_budget_document(budgets)
    expected_layers = int(budget_document["expected_layers"])
    expected_experts = int(budget_document["expected_experts"])
    observations = load_reap_expert_table(
        reap,
        expected_layers=expected_layers,
        expected_experts=expected_experts,
        expected_top_k=int(budget_document["expected_top_k"]),
    )
    raw_candidates = _load_candidates(candidates)
    by_layer: dict[int, list[ExpertProfileEvaluation]] = defaultdict(list)
    for item in raw_candidates:
        key = (item.layer, item.expert)
        try:
            observation = observations[key]
        except KeyError as exc:
            raise ValueError(
                f"candidate layer/expert pair is absent from REAP: {key}"
            ) from exc
        by_layer[item.layer].append(
            replace(
                item,
                exposure=observation.exposure,
                normalized_exposure=observation.normalized_exposure,
            )
        )
    if set(by_layer) != set(range(expected_layers)):
        raise ValueError("candidate table does not cover every budgeted layer")

    source_metadata = {
        "raw_reap": str(reap),
        "raw_reap_sha256": _sha256(reap),
        "candidate_table": str(candidates),
        "candidate_table_sha256": _sha256(candidates),
        "layer_budgets": str(budgets),
        "layer_budgets_sha256": _sha256(budgets),
    }
    expert_selections = {}
    layer_reports: dict[str, Any] = {}
    total_target = 0
    total_actual = 0
    total_loss = 0.0
    selected_counts = {"gate": Counter(), "down": Counter()}
    for layer in range(expected_layers):
        target_bits = int(
            budget_document["layers"][str(layer)]["target_storage_bits"]
        )
        layer_scheme, layer_report = allocate_independent_expert_profiles(
            by_layer[layer],
            target_storage_bits=target_bits,
            target_label=f"{target_label}_LAYER_{layer}",
            metadata={
                **source_metadata,
                "layer": layer,
                "budget_source": "explicit per-layer routed payload bits",
            },
        )
        layer_selections = layer_scheme.expert_selections
        if tensor_namespace == "gguf":
            remapped = {}
            for source_name, selection in layer_selections.items():
                target_name = _hf_to_gguf_name(source_name)
                if target_name is None:
                    raise ValueError(
                        f"no GGUF tensor mapping for {source_name!r}"
                    )
                if target_name in remapped:
                    raise ValueError(
                        f"duplicate GGUF tensor mapping {target_name!r}"
                    )
                remapped[target_name] = replace(
                    selection,
                    name=target_name,
                )
            layer_selections = remapped
        overlap = set(expert_selections) & set(layer_selections)
        if overlap:
            raise RuntimeError(
                f"duplicate expert tensor names across layers: {sorted(overlap)}"
            )
        expert_selections.update(layer_selections)
        actual_bits = int(layer_scheme.storage_bits)
        if actual_bits > target_bits:
            raise RuntimeError(
                f"layer {layer} exceeds its budget: {actual_bits} > {target_bits}"
            )
        total_target += target_bits
        total_actual += actual_bits
        total_loss += float(layer_report["selected_loss"])
        for kind in ("gate", "down"):
            selected_counts[kind].update(
                layer_report["selected_counts"][kind]
            )
        layer_reports[str(layer)] = {
            "target_storage_bits": target_bits,
            "actual_storage_bits": actual_bits,
            "unused_storage_bits": target_bits - actual_bits,
            "storage_utilization": actual_bits / target_bits,
            "source_types": budget_document["layers"][str(layer)].get(
                "source_types", {}
            ),
            "selected_loss": layer_report["selected_loss"],
            "selected_counts": layer_report["selected_counts"],
        }

    scheme = CalibrationScheme(
        path=None,
        target_profile=f"REAP_EW_{target_label}_PER_LAYER_BPW",
        target_storage_bits=total_target,
        selections={},
        metadata={
            "method": "independent per-layer scipy.optimize.milp/HiGHS",
            "objective": (
                "raw REAP normalized exposure times exact NINT weight NMSE"
            ),
            "budget": "each routed layer is independently capped",
            "target_label": target_label,
            "tensor_namespace": tensor_namespace,
            **source_metadata,
        },
        candidate_table={},
        expert_selections=expert_selections,
    )
    if scheme.storage_bits != total_actual:
        raise RuntimeError(
            f"merged scheme storage mismatch: {scheme.storage_bits} != {total_actual}"
        )
    output_scheme.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    save_scheme(output_scheme, scheme)
    report = {
        "format": "mfq.reap-layer-budget-allocation.v1",
        "solver": (
            f"{expected_layers} independent scipy.optimize.milp/HiGHS problems"
        ),
        "expected_layers": expected_layers,
        "expected_experts": expected_experts,
        "target_label": target_label,
        "tensor_namespace": tensor_namespace,
        **source_metadata,
        "target_storage_bits": total_target,
        "actual_storage_bits": total_actual,
        "unused_storage_bits": total_target - total_actual,
        "storage_utilization": total_actual / total_target,
        "selected_loss": total_loss,
        "selected_counts": {
            kind: dict(sorted(counts.items()))
            for kind, counts in selected_counts.items()
        },
        "layer_reports": layer_reports,
        "scheme": str(output_scheme),
        "scheme_storage_bits": scheme.storage_bits,
        "scheme_bpw": scheme.bpw,
    }
    _atomic_json(output_report, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reap", required=True)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--layer-budgets", required=True)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--output-scheme", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument(
        "--tensor-namespace",
        choices=("gguf", "huggingface"),
        default="gguf",
    )
    args = parser.parse_args()
    report = allocate_reap_layer_budget(
        reap_path=args.reap,
        candidate_table_path=args.candidate_table,
        layer_budgets_path=args.layer_budgets,
        output_scheme_path=args.output_scheme,
        output_report_path=args.output_report,
        target_label=args.target_label,
        tensor_namespace=args.tensor_namespace,
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
