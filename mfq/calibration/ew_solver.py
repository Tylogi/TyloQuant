"""Generic expert-wise precision allocation with joint rate and shape constraints.

The solver consumes three versioned control-plane documents:

``mfq.ew-importance.v1``
    Expert scores or ranks.  Scores retain their supplied magnitude unless an
    explicit normalization is requested.  Rank-only inputs use an explicit,
    recorded rank-to-weight policy.

``mfq.ew-candidates.v1``
    Per-expert precision choices with distortion, effective BPW, variable
    storage, per-pool fixed storage, and per-tensor fixed storage.

``mfq.ew-budget.v1``
    Whole-model, projection, and optional layer budgets plus constraints that
    retain peaks and histogram tails from a reference expert-BPW profile.

The selected assignment is emitted as ``CalibrationScheme`` v3 so the normal
MFQ quantizers can consume it without adding another quantization path.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mfq.calibration.artifact import (
    CalibrationScheme,
    ExpertPrecision,
    ExpertSelection,
    ExpertTensorSelection,
)
from mfq.formats.nint import NintSpec

IMPORTANCE_FORMAT = "mfq.ew-importance.v1"
CANDIDATE_FORMAT = "mfq.ew-candidates.v1"
BUDGET_FORMAT = "mfq.ew-budget.v1"
ALLOCATION_FORMAT = "mfq.ew-allocation.v1"

_SCORE_NORMALIZATIONS = {
    "none",
    "global_sum",
    "layer_sum",
    "layer_projection_sum",
    "tensor_sum",
}
_RANK_WEIGHTINGS = {"linear_percentile", "reciprocal", "uniform"}


def _finite_float(value: Any, label: str, *, nonnegative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        condition = "finite and non-negative" if nonnegative else "finite"
        raise ValueError(f"{label} must be {condition}") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        condition = "finite and non-negative" if nonnegative else "finite"
        raise ValueError(f"{label} must be {condition}")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        result = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result <= 0 or not math.isfinite(numeric) or numeric != result:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if result < 0 or not math.isfinite(numeric) or numeric != result:
        raise ValueError(f"{label} must be a non-negative integer")
    return result


@dataclass(frozen=True, order=True)
class EwItemKey:
    """One independently allocated expert projection."""

    tensor: str
    layer: int
    projection: str
    expert_id: int


@dataclass(frozen=True)
class ImportanceEntry:
    layer: int
    expert_id: int
    projection: str | None
    tensor: str | None
    score: float | None
    rank: int | None

    @property
    def scope_key(self) -> tuple[str, object]:
        if self.tensor is not None:
            return ("tensor", self.tensor)
        if self.projection is not None:
            return ("layer_projection", (self.layer, self.projection))
        return ("layer", self.layer)


@dataclass(frozen=True)
class ImportanceTable:
    mode: str
    entries: tuple[ImportanceEntry, ...]
    score_normalization: str
    rank_weighting: str
    metadata: Mapping[str, Any]

    def weights_for(self, items: Sequence[EwItemKey]) -> dict[EwItemKey, float]:
        exact: dict[tuple[str, int], ImportanceEntry] = {}
        projected: dict[tuple[int, str, int], ImportanceEntry] = {}
        shared: dict[tuple[int, int], ImportanceEntry] = {}
        scope_sizes = Counter(entry.scope_key for entry in self.entries)
        for entry in self.entries:
            if entry.tensor is not None:
                exact[(entry.tensor, entry.expert_id)] = entry
            elif entry.projection is not None:
                projected[(entry.layer, entry.projection, entry.expert_id)] = entry
            else:
                shared[(entry.layer, entry.expert_id)] = entry

        raw: dict[EwItemKey, float] = {}
        for item in items:
            entry = exact.get((item.tensor, item.expert_id))
            if entry is None:
                entry = projected.get((item.layer, item.projection, item.expert_id))
            if entry is None:
                entry = shared.get((item.layer, item.expert_id))
            if entry is None:
                raise ValueError(
                    f"importance table does not cover {item.tensor}/expert.{item.expert_id}"
                )
            if entry.layer != item.layer:
                raise ValueError(
                    f"importance tensor entry {entry.tensor!r} has layer "
                    f"{entry.layer}; candidate uses layer {item.layer}"
                )
            if entry.projection is not None and entry.projection != item.projection:
                raise ValueError(
                    f"importance tensor entry {entry.tensor!r} has projection "
                    f"{entry.projection!r}; candidate uses {item.projection!r}"
                )
            if self.mode == "score":
                assert entry.score is not None
                value = entry.score
            else:
                assert entry.rank is not None
                count = scope_sizes[entry.scope_key]
                if self.rank_weighting == "linear_percentile":
                    value = (count - entry.rank + 1) / count
                elif self.rank_weighting == "reciprocal":
                    value = 1.0 / entry.rank
                else:
                    value = 1.0
            raw[item] = float(value)

        if self.score_normalization == "none":
            return raw
        grouped: dict[object, list[EwItemKey]] = defaultdict(list)
        for item in items:
            if self.score_normalization == "global_sum":
                group: object = "global"
            elif self.score_normalization == "layer_sum":
                group = item.layer
            elif self.score_normalization == "layer_projection_sum":
                group = (item.layer, item.projection)
            else:
                group = item.tensor
            grouped[group].append(item)
        normalized = dict(raw)
        for group, members in grouped.items():
            total = sum(raw[item] for item in members)
            if total <= 0:
                raise ValueError(f"importance normalization group {group!r} sums to zero")
            for item in members:
                normalized[item] = raw[item] / total
        return normalized


def load_importance_document(document: Mapping[str, Any]) -> ImportanceTable:
    if document.get("format") != IMPORTANCE_FORMAT:
        raise ValueError(f"unsupported EW importance format: {document.get('format')!r}")
    mode = str(document.get("mode", ""))
    if mode not in {"score", "rank"}:
        raise ValueError("EW importance mode must be 'score' or 'rank'")
    normalization = str(document.get("score_normalization", "none"))
    if normalization not in _SCORE_NORMALIZATIONS:
        raise ValueError(f"unsupported score normalization: {normalization!r}")
    rank_weighting = str(document.get("rank_weighting", "linear_percentile"))
    if rank_weighting not in _RANK_WEIGHTINGS:
        raise ValueError(f"unsupported rank weighting: {rank_weighting!r}")
    raw_entries = document.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("EW importance document must contain a non-empty entries list")

    entries: list[ImportanceEntry] = []
    identities: set[tuple[int, int, str | None, str | None]] = set()
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, Mapping):
            raise ValueError(f"importance entry {index} must be an object")
        layer = _nonnegative_int(raw.get("layer"), f"importance entry {index} layer")
        expert = _nonnegative_int(raw.get("expert_id"), f"importance entry {index} expert_id")
        projection = raw.get("projection")
        projection = None if projection is None else str(projection)
        tensor = raw.get("tensor")
        tensor = None if tensor is None else str(tensor)
        if projection == "" or tensor == "":
            raise ValueError(f"importance entry {index} has an empty scope field")
        identity = (layer, expert, projection, tensor)
        if identity in identities:
            raise ValueError(f"duplicate importance entry: {identity}")
        identities.add(identity)
        if mode == "score":
            if "score" not in raw or "rank" in raw:
                raise ValueError(f"score-mode importance entry {index} must contain only score")
            score = _finite_float(raw["score"], f"importance entry {index} score", nonnegative=True)
            rank = None
        else:
            if "rank" not in raw or "score" in raw:
                raise ValueError(f"rank-mode importance entry {index} must contain only rank")
            rank = _positive_int(raw["rank"], f"importance entry {index} rank")
            score = None
        entries.append(
            ImportanceEntry(
                layer=layer,
                expert_id=expert,
                projection=projection,
                tensor=tensor,
                score=score,
                rank=rank,
            )
        )

    if mode == "score" and not any((entry.score or 0) > 0 for entry in entries):
        raise ValueError("score-mode importance table contains no positive score")
    if mode == "rank":
        ranks_by_scope: dict[tuple[str, object], list[int]] = defaultdict(list)
        for entry in entries:
            assert entry.rank is not None
            ranks_by_scope[entry.scope_key].append(entry.rank)
        for scope, ranks in ranks_by_scope.items():
            expected = list(range(1, len(ranks) + 1))
            if sorted(ranks) != expected:
                raise ValueError(f"rank scope {scope!r} must contain each rank in [1,{len(ranks)}]")
    metadata = document.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("importance metadata must be an object")
    return ImportanceTable(
        mode=mode,
        entries=tuple(entries),
        score_normalization=normalization,
        rank_weighting=rank_weighting,
        metadata=dict(metadata),
    )


def load_importance_table(path: str | Path) -> ImportanceTable:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"EW importance table does not exist: {source}")
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("EW importance document must be an object")
    return load_importance_document(document)


@dataclass(frozen=True)
class EwTensorSpec:
    name: str
    group: str
    layer: int
    projection: str
    n_experts: int
    rows_per_expert: int
    columns: int
    fixed_storage_bits: int
    reference_bpw: tuple[float, ...] | None

    @property
    def weight_count(self) -> int:
        return self.n_experts * self.rows_per_expert * self.columns


@dataclass(frozen=True)
class EwCandidate:
    key: EwItemKey
    profile: str
    precision: ExpertPrecision
    variable_storage_bits: int
    pool_key: str
    pool_storage_bits: int
    distortion: float
    validation_distortion: float
    effective_bpw: float


@dataclass(frozen=True)
class EwCandidateTable:
    tensors: Mapping[str, EwTensorSpec]
    candidates: tuple[EwCandidate, ...]
    artifact_root: Path | None
    metadata: Mapping[str, Any]

    @property
    def items(self) -> tuple[EwItemKey, ...]:
        return tuple(sorted({candidate.key for candidate in self.candidates}))

    @property
    def routed_weight_count(self) -> int:
        return sum(tensor.weight_count for tensor in self.tensors.values())


def _precision_from_document(raw: Mapping[str, Any], label: str) -> ExpertPrecision:
    if "family" not in raw:
        raise ValueError(f"{label} precision has no family")
    nint_raw = raw.get("nint_spec")
    if nint_raw is not None and not isinstance(nint_raw, Mapping):
        raise ValueError(f"{label} nint_spec must be an object")
    options = raw.get("options", {})
    if not isinstance(options, Mapping):
        raise ValueError(f"{label} precision options must be an object")
    return ExpertPrecision(
        family=str(raw["family"]),
        nint_spec=None if nint_raw is None else NintSpec(**dict(nint_raw)),
        artifact=None if raw.get("artifact") is None else str(raw["artifact"]),
        options=tuple((str(key), value) for key, value in options.items()),
    )


def _default_pool_key(precision: ExpertPrecision) -> str:
    value = {
        "family": precision.family,
        "nint_spec": (
            None
            if precision.nint_spec is None
            else {
                "bits": precision.nint_spec.bits,
                "groupsize": precision.nint_spec.groupsize,
                "sub_bits": precision.nint_spec.sub_bits,
            }
        ),
        "artifact": precision.artifact,
        "options": dict(precision.options),
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_candidate_document(
    document: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> EwCandidateTable:
    if document.get("format") != CANDIDATE_FORMAT:
        raise ValueError(f"unsupported EW candidate format: {document.get('format')!r}")
    raw_tensors = document.get("tensors")
    if not isinstance(raw_tensors, list) or not raw_tensors:
        raise ValueError("EW candidate document must contain a non-empty tensors list")
    tensors: dict[str, EwTensorSpec] = {}
    candidates: list[EwCandidate] = []
    for tensor_index, raw_tensor in enumerate(raw_tensors):
        if not isinstance(raw_tensor, Mapping):
            raise ValueError(f"candidate tensor {tensor_index} must be an object")
        name = str(raw_tensor.get("name", ""))
        if not name or name in tensors:
            raise ValueError(f"invalid or duplicate candidate tensor name: {name!r}")
        group = str(raw_tensor.get("group", name))
        projection = str(raw_tensor.get("projection", ""))
        if not group or not projection:
            raise ValueError(f"candidate tensor {name!r} needs group and projection")
        n_experts = _positive_int(raw_tensor.get("n_experts"), f"{name} n_experts")
        rows = _positive_int(raw_tensor.get("rows_per_expert"), f"{name} rows_per_expert")
        columns = _positive_int(raw_tensor.get("columns"), f"{name} columns")
        reference_raw = raw_tensor.get("reference_bpw")
        reference: tuple[float, ...] | None
        if reference_raw is None:
            reference = None
        else:
            if not isinstance(reference_raw, list) or len(reference_raw) != n_experts:
                raise ValueError(
                    f"candidate tensor {name!r} reference_bpw must contain {n_experts} values"
                )
            reference = tuple(
                _finite_float(value, f"{name} reference_bpw", nonnegative=True)
                for value in reference_raw
            )
        tensor = EwTensorSpec(
            name=name,
            group=group,
            layer=_nonnegative_int(raw_tensor.get("layer"), f"{name} layer"),
            projection=projection,
            n_experts=n_experts,
            rows_per_expert=rows,
            columns=columns,
            fixed_storage_bits=_nonnegative_int(
                raw_tensor.get("fixed_storage_bits", 0), f"{name} fixed_storage_bits"
            ),
            reference_bpw=reference,
        )
        tensors[name] = tensor
        raw_candidates = raw_tensor.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError(f"candidate tensor {name!r} contains no candidates")
        profiles_by_expert: dict[int, set[str]] = defaultdict(set)
        pool_specs: dict[str, tuple[int, ExpertPrecision]] = {}
        for candidate_index, raw in enumerate(raw_candidates):
            if not isinstance(raw, Mapping):
                raise ValueError(f"candidate {name}/{candidate_index} must be an object")
            expert = _nonnegative_int(
                raw.get("expert_id"), f"candidate {name}/{candidate_index} expert_id"
            )
            if expert >= n_experts:
                raise ValueError(f"candidate {name}/{candidate_index} expert_id is out of range")
            profile = str(raw.get("profile", ""))
            if not profile or profile in profiles_by_expert[expert]:
                raise ValueError(
                    f"invalid or duplicate candidate profile {name}/{expert}/{profile!r}"
                )
            profiles_by_expert[expert].add(profile)
            precision_raw = raw.get("precision")
            if not isinstance(precision_raw, Mapping):
                raise ValueError(f"candidate {name}/{expert}/{profile} needs precision")
            precision = _precision_from_document(
                precision_raw, f"candidate {name}/{expert}/{profile}"
            )
            pool_key = str(raw.get("pool_key", _default_pool_key(precision)))
            if not pool_key:
                raise ValueError(f"candidate {name}/{expert}/{profile} has empty pool_key")
            pool_bits = _nonnegative_int(
                raw.get("pool_storage_bits", 0),
                f"candidate {name}/{expert}/{profile} pool_storage_bits",
            )
            pool_spec = (pool_bits, precision)
            if pool_key in pool_specs and pool_specs[pool_key] != pool_spec:
                raise ValueError(
                    f"candidate tensor {name!r} pool {pool_key!r} has inconsistent "
                    "storage or precision"
                )
            pool_specs[pool_key] = pool_spec
            distortion = _finite_float(
                raw.get("distortion"),
                f"candidate {name}/{expert}/{profile} distortion",
                nonnegative=True,
            )
            validation = _finite_float(
                raw.get("validation_distortion", distortion),
                f"candidate {name}/{expert}/{profile} validation_distortion",
                nonnegative=True,
            )
            candidates.append(
                EwCandidate(
                    key=EwItemKey(name, tensor.layer, projection, expert),
                    profile=profile,
                    precision=precision,
                    variable_storage_bits=_positive_int(
                        raw.get("variable_storage_bits"),
                        f"candidate {name}/{expert}/{profile} variable_storage_bits",
                    ),
                    pool_key=pool_key,
                    pool_storage_bits=pool_bits,
                    distortion=distortion,
                    validation_distortion=validation,
                    effective_bpw=_finite_float(
                        raw.get("effective_bpw"),
                        f"candidate {name}/{expert}/{profile} effective_bpw",
                        nonnegative=True,
                    ),
                )
            )
        if set(profiles_by_expert) != set(range(n_experts)):
            missing = sorted(set(range(n_experts)) - set(profiles_by_expert))
            raise ValueError(f"candidate tensor {name!r} misses experts {missing[:8]}")

    artifact_root_raw = document.get("artifact_root")
    artifact_root: Path | None = None
    if artifact_root_raw is not None:
        candidate_root = (
            Path(source_path).resolve().parent if source_path is not None else Path.cwd()
        )
        raw_path = Path(str(artifact_root_raw))
        artifact_root = (
            raw_path.resolve() if raw_path.is_absolute() else (candidate_root / raw_path).resolve()
        )
    elif source_path is not None:
        artifact_root = Path(source_path).resolve().parent
    metadata = document.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("candidate metadata must be an object")
    return EwCandidateTable(
        tensors=tensors,
        candidates=tuple(candidates),
        artifact_root=artifact_root,
        metadata=dict(metadata),
    )


def load_candidate_table(path: str | Path) -> EwCandidateTable:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"EW candidate table does not exist: {source}")
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("EW candidate document must be an object")
    return load_candidate_document(document, source_path=source)


@dataclass(frozen=True)
class RateBounds:
    min_bits: int | None = None
    max_bits: int | None = None
    min_bpw: float | None = None
    max_bpw: float | None = None

    def resolve(self, weight_count: int, label: str) -> tuple[int | None, int | None]:
        if weight_count <= 0:
            raise ValueError(f"{label} weight count must be positive")
        minimums = [] if self.min_bits is None else [self.min_bits]
        maximums = [] if self.max_bits is None else [self.max_bits]
        if self.min_bpw is not None:
            minimums.append(int(math.ceil(self.min_bpw * weight_count - 1e-12)))
        if self.max_bpw is not None:
            maximums.append(int(math.floor(self.max_bpw * weight_count + 1e-12)))
        minimum = max(minimums) if minimums else None
        maximum = min(maximums) if maximums else None
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"{label} minimum budget exceeds maximum budget")
        return minimum, maximum


@dataclass(frozen=True)
class HistogramConstraint:
    side: str
    threshold_bpw: float
    relative_tolerance: float
    absolute_tolerance: int


@dataclass(frozen=True)
class ShapeConstraint:
    name: str
    projection: str | None
    layer: int | None
    peak_reference_min_bpw: float | None
    peak_selected_min_bpw: float | None
    min_peak_retention: float
    min_contrast_ratio: float | None
    histogram: tuple[HistogramConstraint, ...]


@dataclass(frozen=True)
class EwBudget:
    target_profile: str
    model_weight_count: int
    model_fixed_storage_bits: int
    total: RateBounds
    projections: Mapping[str, RateBounds]
    layers: Mapping[int, RateBounds]
    shape_constraints: tuple[ShapeConstraint, ...]
    metadata: Mapping[str, Any]


def _bounds_from_document(raw: Mapping[str, Any], label: str) -> RateBounds:
    has_target_bits = "target_bits" in raw
    has_target_bpw = "target_bpw" in raw
    if (has_target_bits or has_target_bpw) and any(
        key in raw for key in ("min_bits", "max_bits", "min_bpw", "max_bpw")
    ):
        raise ValueError(f"{label} cannot mix target and min/max budget fields")
    if has_target_bits and has_target_bpw:
        raise ValueError(f"{label} cannot contain both target_bits and target_bpw")
    if has_target_bits:
        target = _nonnegative_int(raw["target_bits"], f"{label} target_bits")
        tolerance = _nonnegative_int(raw.get("tolerance_bits", 0), f"{label} tolerance_bits")
        return RateBounds(min_bits=max(0, target - tolerance), max_bits=target + tolerance)
    if has_target_bpw:
        target = _finite_float(raw["target_bpw"], f"{label} target_bpw", nonnegative=True)
        tolerance = _finite_float(
            raw.get("tolerance_bpw", 0.0), f"{label} tolerance_bpw", nonnegative=True
        )
        return RateBounds(min_bpw=max(0.0, target - tolerance), max_bpw=target + tolerance)
    minimum_bits = (
        None
        if raw.get("min_bits") is None
        else _nonnegative_int(raw["min_bits"], f"{label} min_bits")
    )
    maximum_bits = (
        None
        if raw.get("max_bits") is None
        else _nonnegative_int(raw["max_bits"], f"{label} max_bits")
    )
    minimum_bpw = (
        None
        if raw.get("min_bpw") is None
        else _finite_float(raw["min_bpw"], f"{label} min_bpw", nonnegative=True)
    )
    maximum_bpw = (
        None
        if raw.get("max_bpw") is None
        else _finite_float(raw["max_bpw"], f"{label} max_bpw", nonnegative=True)
    )
    if all(value is None for value in (minimum_bits, maximum_bits, minimum_bpw, maximum_bpw)):
        raise ValueError(f"{label} contains no budget bound")
    return RateBounds(minimum_bits, maximum_bits, minimum_bpw, maximum_bpw)


def load_budget_document(document: Mapping[str, Any]) -> EwBudget:
    if document.get("format") != BUDGET_FORMAT:
        raise ValueError(f"unsupported EW budget format: {document.get('format')!r}")
    total_raw = document.get("total")
    if not isinstance(total_raw, Mapping):
        raise ValueError("EW budget document needs a total budget object")
    projections_raw = document.get("projections", {})
    layers_raw = document.get("layers", {})
    if not isinstance(projections_raw, Mapping) or not isinstance(layers_raw, Mapping):
        raise ValueError("EW projection and layer budgets must be objects")
    projections = {
        str(projection): _bounds_from_document(raw, f"projection {projection}")
        for projection, raw in projections_raw.items()
        if isinstance(raw, Mapping)
    }
    if len(projections) != len(projections_raw):
        raise ValueError("every EW projection budget must be an object")
    layers: dict[int, RateBounds] = {}
    for raw_layer, raw in layers_raw.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"layer budget {raw_layer!r} must be an object")
        layer = _nonnegative_int(raw_layer, f"layer budget key {raw_layer!r}")
        if layer in layers:
            raise ValueError(f"duplicate layer budget {layer}")
        layers[layer] = _bounds_from_document(raw, f"layer {layer}")

    raw_shapes = document.get("shape_constraints", [])
    if not isinstance(raw_shapes, list):
        raise ValueError("shape_constraints must be a list")
    shapes: list[ShapeConstraint] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_shapes):
        if not isinstance(raw, Mapping):
            raise ValueError(f"shape constraint {index} must be an object")
        name = str(raw.get("name", ""))
        if not name or name in names:
            raise ValueError(f"invalid or duplicate shape constraint name: {name!r}")
        names.add(name)
        projection = raw.get("projection")
        projection = None if projection is None else str(projection)
        layer = raw.get("layer")
        layer = None if layer is None else _nonnegative_int(layer, f"shape {name} layer")
        peak_reference = raw.get("peak_reference_min_bpw")
        peak_reference = (
            None
            if peak_reference is None
            else _finite_float(
                peak_reference, f"shape {name} peak_reference_min_bpw", nonnegative=True
            )
        )
        peak_selected = raw.get("peak_selected_min_bpw")
        peak_selected = (
            None
            if peak_selected is None
            else _finite_float(
                peak_selected, f"shape {name} peak_selected_min_bpw", nonnegative=True
            )
        )
        retention = _finite_float(
            raw.get("min_peak_retention", 0.0),
            f"shape {name} min_peak_retention",
            nonnegative=True,
        )
        if retention > 1:
            raise ValueError(f"shape {name} min_peak_retention must not exceed 1")
        contrast = raw.get("min_contrast_ratio")
        contrast = (
            None
            if contrast is None
            else _finite_float(contrast, f"shape {name} min_contrast_ratio", nonnegative=True)
        )
        if retention > 0 and (peak_reference is None or peak_selected is None):
            raise ValueError(
                f"shape {name} peak retention requires reference and selected thresholds"
            )
        if contrast is not None and peak_reference is None:
            raise ValueError(f"shape {name} contrast requires peak_reference_min_bpw")
        raw_histogram = raw.get("histogram", [])
        if not isinstance(raw_histogram, list):
            raise ValueError(f"shape {name} histogram must be a list")
        histogram: list[HistogramConstraint] = []
        for histogram_index, histogram_raw in enumerate(raw_histogram):
            if not isinstance(histogram_raw, Mapping):
                raise ValueError(f"shape {name} histogram {histogram_index} must be an object")
            side = str(histogram_raw.get("side", ""))
            if side not in {"le", "ge"}:
                raise ValueError(f"shape {name} histogram side must be 'le' or 'ge'")
            relative = _finite_float(
                histogram_raw.get("relative_tolerance", 0.0),
                f"shape {name} histogram relative_tolerance",
                nonnegative=True,
            )
            histogram.append(
                HistogramConstraint(
                    side=side,
                    threshold_bpw=_finite_float(
                        histogram_raw.get("threshold_bpw"),
                        f"shape {name} histogram threshold_bpw",
                        nonnegative=True,
                    ),
                    relative_tolerance=relative,
                    absolute_tolerance=_nonnegative_int(
                        histogram_raw.get("absolute_tolerance", 0),
                        f"shape {name} histogram absolute_tolerance",
                    ),
                )
            )
        if retention == 0 and contrast is None and not histogram:
            raise ValueError(f"shape constraint {name!r} contains no active constraint")
        shapes.append(
            ShapeConstraint(
                name=name,
                projection=projection,
                layer=layer,
                peak_reference_min_bpw=peak_reference,
                peak_selected_min_bpw=peak_selected,
                min_peak_retention=retention,
                min_contrast_ratio=contrast,
                histogram=tuple(histogram),
            )
        )
    metadata = document.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("budget metadata must be an object")
    return EwBudget(
        target_profile=str(document.get("target_profile", "EW-JOINT")),
        model_weight_count=_positive_int(document.get("model_weight_count"), "model_weight_count"),
        model_fixed_storage_bits=_nonnegative_int(
            document.get("model_fixed_storage_bits", 0), "model_fixed_storage_bits"
        ),
        total=_bounds_from_document(total_raw, "total"),
        projections=projections,
        layers=layers,
        shape_constraints=tuple(shapes),
        metadata=dict(metadata),
    )


def load_budget(path: str | Path) -> EwBudget:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"EW budget document does not exist: {source}")
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("EW budget document must be an object")
    return load_budget_document(document)


@dataclass(frozen=True)
class EwSolveResult:
    scheme: CalibrationScheme
    report: Mapping[str, Any]
    selected: Mapping[EwItemKey, EwCandidate]


@dataclass(frozen=True)
class _BudgetRow:
    label: str
    row: int
    scale: float
    minimum: int | None
    maximum: int | None
    constant_bits: int
    x_indices: tuple[int, ...]
    pool_indices: tuple[int, ...]


def _scope_cost(
    selected_indices: set[int],
    active_pool_indices: set[int],
    row: _BudgetRow,
    candidates: Sequence[EwCandidate],
    pool_bits: Sequence[int],
) -> int:
    return int(
        row.constant_bits
        + sum(
            candidates[index].variable_storage_bits
            for index in row.x_indices
            if index in selected_indices
        )
        + sum(pool_bits[index] for index in row.pool_indices if index in active_pool_indices)
    )


def solve_ew_budget(
    importance: ImportanceTable,
    candidate_table: EwCandidateTable,
    budget: EwBudget,
) -> EwSolveResult:
    """Solve the joint expert allocation exactly with SciPy/HiGHS MILP."""

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("EW joint allocation requires scipy") from exc

    candidates = list(candidate_table.candidates)
    items = candidate_table.items
    if not candidates or not items:
        raise ValueError("EW candidate table is empty")
    if budget.model_weight_count < candidate_table.routed_weight_count:
        raise ValueError(
            "model_weight_count is smaller than the routed candidate weight count: "
            f"{budget.model_weight_count} < {candidate_table.routed_weight_count}"
        )
    importance_weights = importance.weights_for(items)
    by_item: dict[EwItemKey, list[int]] = defaultdict(list)
    by_pool: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_item[candidate.key].append(index)
        by_pool[(candidate.key.tensor, candidate.pool_key)].append(index)

    pool_keys = sorted(by_pool)
    pool_offset = len(candidates)
    pool_variable = {key: pool_offset + index for index, key in enumerate(pool_keys)}
    pool_local_index = {key: index for index, key in enumerate(pool_keys)}
    pool_bits = [candidates[by_pool[key][0]].pool_storage_bits for key in pool_keys]
    variable_count = len(candidates) + len(pool_keys)

    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    labels: list[str] = []

    def add_row(
        entries: Iterable[tuple[int, float]],
        minimum: float,
        maximum: float,
        label: str,
    ) -> int:
        row = len(lower)
        for column, value in entries:
            rows.append(row)
            columns.append(column)
            data.append(float(value))
        lower.append(float(minimum))
        upper.append(float(maximum))
        labels.append(label)
        return row

    for item in items:
        add_row(
            ((index, 1.0) for index in by_item[item]),
            1.0,
            1.0,
            f"choice:{item.tensor}:{item.expert_id}",
        )
    for pool_key in pool_keys:
        indices = by_pool[pool_key]
        y = pool_variable[pool_key]
        add_row(
            [*((index, 1.0) for index in indices), (y, -float(len(indices)))],
            -np.inf,
            0.0,
            f"pool-enable:{pool_key}",
        )
        add_row(
            [(y, 1.0), *((index, -1.0) for index in indices)],
            -np.inf,
            0.0,
            f"pool-disable:{pool_key}",
        )

    budget_rows: list[_BudgetRow] = []

    def add_budget_row(
        label: str,
        selected_tensors: set[str],
        bounds: RateBounds,
        weight_count: int,
        external_fixed_bits: int = 0,
    ) -> None:
        minimum, maximum = bounds.resolve(weight_count, label)
        x_indices = tuple(
            index
            for index, candidate in enumerate(candidates)
            if candidate.key.tensor in selected_tensors
        )
        selected_pools = tuple(
            pool_local_index[key] for key in pool_keys if key[0] in selected_tensors
        )
        tensor_fixed = sum(
            candidate_table.tensors[name].fixed_storage_bits for name in selected_tensors
        )
        constant = int(external_fixed_bits + tensor_fixed)
        variable_max = sum(
            max(candidates[index].variable_storage_bits for index in by_item[item])
            for item in items
            if item.tensor in selected_tensors
        ) + sum(pool_bits[index] for index in selected_pools)
        scale = float(max(1, maximum or 0, minimum or 0, constant + variable_max))
        entries: list[tuple[int, float]] = [
            (index, candidates[index].variable_storage_bits / scale) for index in x_indices
        ]
        entries.extend((pool_offset + index, pool_bits[index] / scale) for index in selected_pools)
        row = add_row(
            entries,
            -np.inf if minimum is None else (minimum - constant) / scale,
            np.inf if maximum is None else (maximum - constant) / scale,
            f"budget:{label}",
        )
        budget_rows.append(
            _BudgetRow(
                label=label,
                row=row,
                scale=scale,
                minimum=minimum,
                maximum=maximum,
                constant_bits=constant,
                x_indices=x_indices,
                pool_indices=selected_pools,
            )
        )

    all_tensors = set(candidate_table.tensors)
    add_budget_row(
        "total",
        all_tensors,
        budget.total,
        budget.model_weight_count,
        budget.model_fixed_storage_bits,
    )
    for projection, bounds in sorted(budget.projections.items()):
        selected_tensors = {
            name
            for name, tensor in candidate_table.tensors.items()
            if tensor.projection == projection
        }
        if not selected_tensors:
            raise ValueError(f"projection budget {projection!r} selects no candidate tensor")
        weight_count = sum(candidate_table.tensors[name].weight_count for name in selected_tensors)
        add_budget_row(f"projection:{projection}", selected_tensors, bounds, weight_count)
    for layer, bounds in sorted(budget.layers.items()):
        selected_tensors = {
            name for name, tensor in candidate_table.tensors.items() if tensor.layer == layer
        }
        if not selected_tensors:
            raise ValueError(f"layer budget {layer} selects no candidate tensor")
        weight_count = sum(candidate_table.tensors[name].weight_count for name in selected_tensors)
        add_budget_row(f"layer:{layer}", selected_tensors, bounds, weight_count)

    shape_groups: dict[str, tuple[EwItemKey, ...]] = {}
    for shape in budget.shape_constraints:
        members = tuple(
            item
            for item in items
            if (shape.projection is None or item.projection == shape.projection)
            and (shape.layer is None or item.layer == shape.layer)
        )
        if not members:
            raise ValueError(f"shape constraint {shape.name!r} selects no experts")
        reference: dict[EwItemKey, float] = {}
        for item in members:
            tensor = candidate_table.tensors[item.tensor]
            if tensor.reference_bpw is None:
                raise ValueError(
                    f"shape constraint {shape.name!r} requires reference_bpw for {item.tensor}"
                )
            reference[item] = tensor.reference_bpw[item.expert_id]
        shape_groups[shape.name] = members
        if shape.peak_reference_min_bpw is not None:
            peaks = tuple(
                item for item in members if reference[item] >= shape.peak_reference_min_bpw
            )
            peak_set = set(peaks)
            body = tuple(item for item in members if item not in peak_set)
            if not peaks:
                raise ValueError(f"shape constraint {shape.name!r} has an empty peak set")
            if shape.min_peak_retention > 0:
                assert shape.peak_selected_min_bpw is not None
                entries = [
                    (index, 1.0)
                    for item in peaks
                    for index in by_item[item]
                    if candidates[index].effective_bpw >= shape.peak_selected_min_bpw
                ]
                required = int(math.ceil(shape.min_peak_retention * len(peaks) - 1e-12))
                add_row(entries, float(required), np.inf, f"shape:{shape.name}:peak-retention")
            if shape.min_contrast_ratio is not None:
                if not body:
                    raise ValueError(f"shape constraint {shape.name!r} has an empty body set")
                reference_gap = sum(reference[item] for item in peaks) / len(peaks) - sum(
                    reference[item] for item in body
                ) / len(body)
                if reference_gap <= 0:
                    raise ValueError(
                        f"shape constraint {shape.name!r} reference peak/body gap is non-positive"
                    )
                entries = []
                for item in peaks:
                    entries.extend(
                        (index, candidates[index].effective_bpw / len(peaks))
                        for index in by_item[item]
                    )
                for item in body:
                    entries.extend(
                        (index, -candidates[index].effective_bpw / len(body))
                        for index in by_item[item]
                    )
                add_row(
                    entries,
                    shape.min_contrast_ratio * reference_gap,
                    np.inf,
                    f"shape:{shape.name}:contrast",
                )
        for histogram_index, histogram in enumerate(shape.histogram):
            predicate = (
                (lambda value, threshold=histogram.threshold_bpw: value <= threshold)
                if histogram.side == "le"
                else (lambda value, threshold=histogram.threshold_bpw: value >= threshold)
            )
            reference_count = sum(predicate(reference[item]) for item in members)
            lower_count = max(
                0,
                int(
                    math.ceil(
                        reference_count * (1.0 - histogram.relative_tolerance)
                        - histogram.absolute_tolerance
                        - 1e-12
                    )
                ),
            )
            upper_count = min(
                len(members),
                int(
                    math.floor(
                        reference_count * (1.0 + histogram.relative_tolerance)
                        + histogram.absolute_tolerance
                        + 1e-12
                    )
                ),
            )
            entries = [
                (index, 1.0)
                for item in members
                for index in by_item[item]
                if predicate(candidates[index].effective_bpw)
            ]
            add_row(
                entries,
                float(lower_count),
                float(upper_count),
                f"shape:{shape.name}:histogram:{histogram_index}",
            )

    objective = np.zeros(variable_count, dtype=np.float64)
    for index, candidate in enumerate(candidates):
        objective[index] = importance_weights[candidate.key] * candidate.distortion
    positive = objective[objective > 0]
    objective_scale = float(np.median(positive)) if positive.size else 1.0
    objective /= objective_scale
    cost_scale = max(
        1.0,
        float(sum(tensor.fixed_storage_bits for tensor in candidate_table.tensors.values())),
        float(sum(candidate.variable_storage_bits for candidate in candidates)),
    )
    for index, candidate in enumerate(candidates):
        objective[index] += 1e-12 * candidate.variable_storage_bits / cost_scale
    for local_index, value in enumerate(pool_bits):
        objective[pool_offset + local_index] = 1e-12 * value / cost_scale

    matrix = coo_matrix((data, (rows, columns)), shape=(len(lower), variable_count)).tocsr()
    base_lower = np.asarray(lower, dtype=np.float64)
    base_upper = np.asarray(upper, dtype=np.float64)
    solve_lower = base_lower.copy()
    solve_upper = base_upper.copy()
    integrality = np.ones(variable_count, dtype=np.uint8)
    selected_indices: set[int] = set()
    active_pool_indices: set[int] = set()
    result = None
    for _attempt in range(5):
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
            constraints=LinearConstraint(matrix, solve_lower, solve_upper),
            options={"presolve": True},
        )
        if not result.success or result.x is None:
            raise ValueError(f"EW joint allocation is infeasible: {result.message}")
        selected_indices = {index for index in range(len(candidates)) if result.x[index] > 0.5}
        active_pool_indices = {
            index for index in range(len(pool_keys)) if result.x[pool_offset + index] > 0.5
        }
        violations = []
        for row in budget_rows:
            actual = _scope_cost(selected_indices, active_pool_indices, row, candidates, pool_bits)
            if row.minimum is not None and actual < row.minimum:
                violations.append((row, "minimum", row.minimum - actual))
            if row.maximum is not None and actual > row.maximum:
                violations.append((row, "maximum", actual - row.maximum))
        if not violations:
            break
        for row, direction, delta in violations:
            reference = row.maximum if row.maximum is not None else row.minimum or 1
            guard = max(1, int(math.ceil(reference * 1e-7)))
            if direction == "minimum":
                solve_lower[row.row] += (delta + guard) / row.scale
            else:
                solve_upper[row.row] -= (delta + guard) / row.scale
    else:
        raise RuntimeError("EW joint allocation exceeds an integer budget after tightening")
    assert result is not None

    selected: dict[EwItemKey, EwCandidate] = {}
    for item in items:
        choices = [index for index in by_item[item] if index in selected_indices]
        if len(choices) != 1:
            raise RuntimeError(f"EW MILP selected {len(choices)} choices for {item}")
        selected[item] = candidates[choices[0]]
    selected_pool_keys = {pool_keys[index] for index in active_pool_indices}
    expected_pool_keys = {
        (candidate.key.tensor, candidate.pool_key) for candidate in selected.values()
    }
    if selected_pool_keys != expected_pool_keys:
        raise RuntimeError("EW MILP pool activation differs from selected candidates")

    storage_by_item = {
        item: candidate.variable_storage_bits for item, candidate in selected.items()
    }
    for tensor_name, tensor in candidate_table.tensors.items():
        tensor_items = sorted(item for item in items if item.tensor == tensor_name)
        storage_by_item[tensor_items[0]] += tensor.fixed_storage_bits
        selected_by_pool: dict[str, list[EwItemKey]] = defaultdict(list)
        for item in tensor_items:
            selected_by_pool[selected[item].pool_key].append(item)
        for _pool_key, members in selected_by_pool.items():
            storage_by_item[min(members)] += selected[members[0]].pool_storage_bits

    expert_selections: dict[str, ExpertTensorSelection] = {}
    for tensor_name, tensor in sorted(candidate_table.tensors.items()):
        tensor_items = sorted(
            (item for item in items if item.tensor == tensor_name),
            key=lambda item: item.expert_id,
        )
        expert_selections[tensor_name] = ExpertTensorSelection(
            name=tensor_name,
            group=tensor.group,
            n_experts=tensor.n_experts,
            rows_per_expert=tensor.rows_per_expert,
            columns=tensor.columns,
            selections=tuple(
                ExpertSelection(
                    expert_id=item.expert_id,
                    spec=selected[item].precision.nint_spec,
                    precision=selected[item].precision,
                    storage_bits=storage_by_item[item],
                    train_loss=importance_weights[item] * selected[item].distortion,
                    validation_loss=(
                        importance_weights[item] * selected[item].validation_distortion
                    ),
                )
                for item in tensor_items
            ),
        )

    routed_storage_bits = sum(storage_by_item.values())
    total_minimum, total_maximum = budget.total.resolve(budget.model_weight_count, "total")
    target_routed_bits = (
        routed_storage_bits
        if total_maximum is None
        else total_maximum - budget.model_fixed_storage_bits
    )
    if target_routed_bits < routed_storage_bits:
        raise RuntimeError("resolved routed target is smaller than selected routed storage")
    scheme = CalibrationScheme(
        path=None,
        target_profile=budget.target_profile,
        target_storage_bits=int(target_routed_bits),
        selections={},
        expert_selections=expert_selections,
        metadata={
            "method": "generic-ew-joint-milp",
            "solver": "scipy.optimize.milp/highs",
            "importance_mode": importance.mode,
            "score_normalization": importance.score_normalization,
            "rank_weighting": importance.rank_weighting if importance.mode == "rank" else None,
            "storage_accounting": (
                "per-expert variable bits + active-pool fixed bits + tensor fixed bits"
            ),
            "model_weight_count": budget.model_weight_count,
            "model_fixed_storage_bits": budget.model_fixed_storage_bits,
            "planned_model_storage_bits": budget.model_fixed_storage_bits + routed_storage_bits,
            **dict(budget.metadata),
        },
        candidate_table={},
    )
    if scheme.storage_bits != routed_storage_bits:
        raise RuntimeError(
            f"EW scheme storage mismatch: {scheme.storage_bits} != {routed_storage_bits}"
        )

    budget_report: dict[str, Any] = {}
    for row in budget_rows:
        actual = _scope_cost(selected_indices, active_pool_indices, row, candidates, pool_bits)
        if row.label == "total":
            weight_count = budget.model_weight_count
        elif row.label.startswith("projection:"):
            projection = row.label.split(":", 1)[1]
            weight_count = sum(
                tensor.weight_count
                for tensor in candidate_table.tensors.values()
                if tensor.projection == projection
            )
        else:
            layer = int(row.label.split(":", 1)[1])
            weight_count = sum(
                tensor.weight_count
                for tensor in candidate_table.tensors.values()
                if tensor.layer == layer
            )
        budget_report[row.label] = {
            "minimum_storage_bits": row.minimum,
            "maximum_storage_bits": row.maximum,
            "actual_storage_bits": actual,
            "actual_bpw": actual / weight_count,
            "within_bounds": (
                (row.minimum is None or actual >= row.minimum)
                and (row.maximum is None or actual <= row.maximum)
            ),
        }

    shape_report: dict[str, Any] = {}
    for shape in budget.shape_constraints:
        members = shape_groups[shape.name]
        reference = {
            item: candidate_table.tensors[item.tensor].reference_bpw[item.expert_id]  # type: ignore[index]
            for item in members
        }
        selected_bpw = {item: selected[item].effective_bpw for item in members}
        item_report: dict[str, Any] = {"experts": len(members)}
        if shape.peak_reference_min_bpw is not None:
            peaks = tuple(
                item for item in members if reference[item] >= shape.peak_reference_min_bpw
            )
            peak_set = set(peaks)
            body = tuple(item for item in members if item not in peak_set)
            retained = (
                0
                if shape.peak_selected_min_bpw is None
                else sum(selected_bpw[item] >= shape.peak_selected_min_bpw for item in peaks)
            )
            item_report.update(
                {
                    "peak_experts": len(peaks),
                    "retained_peak_experts": retained,
                    "peak_retention": retained / len(peaks),
                }
            )
            if body:
                reference_contrast = sum(reference[item] for item in peaks) / len(peaks) - sum(
                    reference[item] for item in body
                ) / len(body)
                selected_contrast = sum(selected_bpw[item] for item in peaks) / len(peaks) - sum(
                    selected_bpw[item] for item in body
                ) / len(body)
                item_report.update(
                    {
                        "reference_peak_body_contrast_bpw": reference_contrast,
                        "selected_peak_body_contrast_bpw": selected_contrast,
                        "contrast_ratio": selected_contrast / reference_contrast,
                    }
                )
        histogram_report = []
        for histogram in shape.histogram:
            predicate = (
                (lambda value, threshold=histogram.threshold_bpw: value <= threshold)
                if histogram.side == "le"
                else (lambda value, threshold=histogram.threshold_bpw: value >= threshold)
            )
            reference_count = sum(predicate(reference[item]) for item in members)
            actual_count = sum(predicate(selected_bpw[item]) for item in members)
            lower_count = max(
                0,
                int(
                    math.ceil(
                        reference_count * (1.0 - histogram.relative_tolerance)
                        - histogram.absolute_tolerance
                        - 1e-12
                    )
                ),
            )
            upper_count = min(
                len(members),
                int(
                    math.floor(
                        reference_count * (1.0 + histogram.relative_tolerance)
                        + histogram.absolute_tolerance
                        + 1e-12
                    )
                ),
            )
            histogram_report.append(
                {
                    "side": histogram.side,
                    "threshold_bpw": histogram.threshold_bpw,
                    "reference_count": reference_count,
                    "minimum_count": lower_count,
                    "maximum_count": upper_count,
                    "actual_count": actual_count,
                    "within_bounds": lower_count <= actual_count <= upper_count,
                }
            )
        item_report["histogram"] = histogram_report
        shape_report[shape.name] = item_report

    selected_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate in selected.values():
        selected_counts[candidate.key.projection][candidate.precision.family] += 1
    report = {
        "format": ALLOCATION_FORMAT,
        "solver": "scipy.optimize.milp/highs",
        "status": "optimal",
        "items": len(items),
        "candidates": len(candidates),
        "pools": len(pool_keys),
        "importance": {
            "mode": importance.mode,
            "score_normalization": importance.score_normalization,
            "rank_weighting": importance.rank_weighting if importance.mode == "rank" else None,
        },
        "routed_weight_count": candidate_table.routed_weight_count,
        "routed_storage_bits": routed_storage_bits,
        "routed_bpw": routed_storage_bits / candidate_table.routed_weight_count,
        "model_weight_count": budget.model_weight_count,
        "model_fixed_storage_bits": budget.model_fixed_storage_bits,
        "model_storage_bits": budget.model_fixed_storage_bits + routed_storage_bits,
        "model_bpw": (budget.model_fixed_storage_bits + routed_storage_bits)
        / budget.model_weight_count,
        "objective_train_loss": float(
            sum(
                importance_weights[item] * candidate.distortion
                for item, candidate in selected.items()
            )
        ),
        "objective_validation_loss": float(
            sum(
                importance_weights[item] * candidate.validation_distortion
                for item, candidate in selected.items()
            )
        ),
        "budgets": budget_report,
        "shape_constraints": shape_report,
        "selected_counts": {
            projection: dict(sorted(counts.items()))
            for projection, counts in sorted(selected_counts.items())
        },
    }
    if total_minimum is not None and report["model_storage_bits"] < total_minimum:
        raise RuntimeError("EW allocation report is below the total minimum budget")
    if total_maximum is not None and report["model_storage_bits"] > total_maximum:
        raise RuntimeError("EW allocation report exceeds the total maximum budget")
    return EwSolveResult(scheme=scheme, report=report, selected=selected)


__all__ = [
    "ALLOCATION_FORMAT",
    "BUDGET_FORMAT",
    "CANDIDATE_FORMAT",
    "IMPORTANCE_FORMAT",
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
]
