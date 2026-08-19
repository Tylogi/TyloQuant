"""Calibration data, statistics, allocation, and imatrix APIs."""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Keep the package facade lightweight. Quantizers depend on calibration
# artifact types, so importing every calibration pipeline here would create a
# cycle between ``mfq.quantize`` and ``mfq.calibration``.
_MODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "allocator": ("AllocationResult", "GroupCandidate", "allocate"),
    "artifact": (
        "CalibrationScheme",
        "ExpertPrecision",
        "ExpertSelection",
        "ExpertTensorSelection",
        "TensorSelection",
        "load_scheme",
        "nint_expert_precision",
        "save_scheme",
        "scheme_expert_precisions",
        "scheme_expert_specs",
    ),
    "collector": ("HiddenTrace", "validate_layerwise"),
    "dataset": (
        "CalibrationBatch",
        "CalibrationCorpus",
        "TraceSource",
        "build_corpus_from_records",
        "build_eaddario_corpus",
        "build_hf_trace_corpus",
        "build_trace_corpus_from_jsonl",
        "load_corpus",
    ),
    "evaluator": (
        "LayerStrategy",
        "build_compensated_layer_strategies",
        "build_global_layer_strategies",
        "build_layer_strategies",
        "load_scheme_candidate_evaluations",
    ),
    "ew_solver": (
        "EwBudget",
        "EwCandidate",
        "EwCandidateTable",
        "EwItemKey",
        "EwSolveResult",
        "EwTensorSpec",
        "HistogramConstraint",
        "ImportanceEntry",
        "ImportanceTable",
        "RateBounds",
        "ShapeConstraint",
        "load_budget",
        "load_budget_document",
        "load_candidate_document",
        "load_candidate_table",
        "load_importance_document",
        "load_importance_table",
        "solve_ew_budget",
    ),
    "imatrix": (
        "ActivationImatrixCollector",
        "ImatrixTarget",
        "collect_imatrix",
    ),
    "inint": ("InintSelector", "build_inint_selector", "load_inint_selector"),
    "refinement": (
        "GlobalRefinementAttempt",
        "GlobalRefinementRound",
        "LayerRefinement",
        "refine_layerwise",
        "refine_layerwise_global",
    ),
    "statistics": (
        "CalibrationStatistics",
        "TensorStatistics",
        "collect_qwen35_statistics",
        "load_statistics",
    ),
}

__all__ = [
    "ActivationImatrixCollector",
    "AllocationResult",
    "CalibrationBatch",
    "CalibrationCorpus",
    "CalibrationScheme",
    "CalibrationStatistics",
    "ExpertPrecision",
    "ExpertSelection",
    "ExpertTensorSelection",
    "EwBudget",
    "EwCandidate",
    "EwCandidateTable",
    "EwItemKey",
    "EwSolveResult",
    "EwTensorSpec",
    "GlobalRefinementAttempt",
    "GlobalRefinementRound",
    "GroupCandidate",
    "HiddenTrace",
    "HistogramConstraint",
    "ImatrixTarget",
    "ImportanceEntry",
    "ImportanceTable",
    "InintSelector",
    "LayerRefinement",
    "LayerStrategy",
    "RateBounds",
    "ShapeConstraint",
    "TensorSelection",
    "TensorStatistics",
    "TraceSource",
    "allocate",
    "build_compensated_layer_strategies",
    "build_corpus_from_records",
    "build_eaddario_corpus",
    "build_global_layer_strategies",
    "build_hf_trace_corpus",
    "build_inint_selector",
    "build_layer_strategies",
    "build_trace_corpus_from_jsonl",
    "collect_imatrix",
    "collect_qwen35_statistics",
    "load_budget",
    "load_budget_document",
    "load_candidate_document",
    "load_candidate_table",
    "load_corpus",
    "load_importance_document",
    "load_importance_table",
    "load_inint_selector",
    "load_scheme",
    "load_scheme_candidate_evaluations",
    "load_statistics",
    "nint_expert_precision",
    "refine_layerwise",
    "refine_layerwise_global",
    "save_scheme",
    "scheme_expert_precisions",
    "scheme_expert_specs",
    "solve_ew_budget",
    "validate_layerwise",
]

_EXPORT_MODULES = {
    symbol: f"{__name__}.{module_name}"
    for module_name, symbols in _MODULE_EXPORTS.items()
    for symbol in symbols
}
if set(__all__) != set(_EXPORT_MODULES):  # pragma: no cover - module invariant
    raise RuntimeError("mfq.calibration lazy exports are out of sync with __all__")


def __getattr__(name: str) -> Any:
    if name in _MODULE_EXPORTS:
        value = import_module(f"{__name__}.{name}")
    else:
        module_name = _EXPORT_MODULES.get(name)
        if module_name is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__, *_MODULE_EXPORTS})
