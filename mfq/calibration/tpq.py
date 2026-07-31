"""Routing-energy calibration for TPQ expert precision tiers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from mfq.calibration.artifact import (
    ExpertPrecision,
    ExpertSelection,
    ExpertTensorSelection,
)
from mfq.formats.tpq import CccpPqSpec
from mfq.formats.tpq import TPQ_PQ_SPECS

_TIER_CHARACTER = {
    "x": "x",
    "w": "w",
    "v": "v",
    "V": "vv",
}


@dataclass(frozen=True)
class CccpTierAllocation:
    """One layer's expert tier assignment and captured score mass."""

    tiers: tuple[str, ...]
    scores: tuple[float, ...]
    boundaries: tuple[int, int]
    score_mass: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.tiers or len(self.tiers) != len(self.scores):
            raise ValueError("CCCP allocation must cover every expert")
        if any(tier not in {"x", "w", "v", "vv"} for tier in self.tiers):
            raise ValueError("CCCP allocation contains an unsupported tier")

    @property
    def counts(self) -> dict[str, int]:
        return {
            tier: self.tiers.count(tier)
            for tier in ("vv", "v", "w", "x")
        }


def allocate_cccp_tiers(
    scores: Sequence[float] | np.ndarray,
    *,
    v_coverage: float = 0.965,
    w_coverage: float = 0.997,
    vv_share: float = 0.25,
) -> CccpTierAllocation:
    """Assign vv/v/w/x using CCCP's per-layer cumulative score thresholds."""

    if not 0 < v_coverage < w_coverage <= 1:
        raise ValueError("CCCP coverage must satisfy 0 < v < w <= 1")
    if not 0 < vv_share <= 1:
        raise ValueError("CCCP vv_share must be in (0,1]")
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if (
        values.size == 0
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.any(values > 0)
    ):
        raise ValueError("CCCP scores must be finite, non-negative, and non-zero")
    total = float(values.sum())
    order = np.argsort(-values, kind="stable")
    cumulative = np.cumsum(values[order]) / total
    v_bound = min(values.size, int(np.searchsorted(cumulative, v_coverage)) + 1)
    w_bound = min(values.size, int(np.searchsorted(cumulative, w_coverage)) + 1)
    tiers = np.full(values.size, "x", dtype="<U2")
    tiers[order[:w_bound]] = "w"
    tiers[order[:v_bound]] = "v"
    tiers[values / total >= vv_share] = "vv"
    mass = tuple(
        (
            tier,
            float(values[tiers == tier].sum() / total),
        )
        for tier in ("vv", "v", "w", "x")
    )
    return CccpTierAllocation(
        tiers=tuple(str(value) for value in tiers),
        scores=tuple(float(value) for value in values),
        boundaries=(v_bound, w_bound),
        score_mass=mass,
    )


def load_cccp_score_profile(
    path: str | Path,
    *,
    field: str = "counts",
) -> dict[int, np.ndarray]:
    """Load a layer-to-expert score profile from CCCP-style JSON."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = document.get(field, document)
    if not isinstance(raw, Mapping):
        raise ValueError("CCCP score profile must contain a layer mapping")
    result: dict[int, np.ndarray] = {}
    for layer, expert_scores in raw.items():
        if not isinstance(expert_scores, Mapping) or not expert_scores:
            raise ValueError(f"CCCP layer {layer} has no expert scores")
        expert_ids = sorted(int(value) for value in expert_scores)
        expected = list(range(expert_ids[-1] + 1))
        if expert_ids != expected:
            raise ValueError(f"CCCP layer {layer} expert IDs are not contiguous")
        result[int(layer)] = np.asarray(
            [float(expert_scores[str(expert)]) for expert in expected],
            dtype=np.float64,
        )
    return result


def _load_profile_document(path: str | Path) -> dict:
    text = Path(path).read_text(encoding="utf-8").strip()
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        fragment = text[:-1].rstrip() if text.endswith(",") else text
        try:
            document = json.loads("{" + fragment + "}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid CCCP calibration JSON: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError("CCCP calibration document must be an object")
    return document


def load_cccp_tier_profile(
    path: str | Path,
    *,
    field: str = "tiers_per_layer",
) -> dict[int, tuple[str, ...]] | None:
    """Load a fixed per-layer ``x/w/v/V`` expert tier assignment."""

    document = _load_profile_document(path)
    raw = document.get(field)
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"CCCP {field} must be a non-empty layer mapping")
    result: dict[int, tuple[str, ...]] = {}
    width: int | None = None
    for raw_layer, encoded in raw.items():
        if not isinstance(encoded, str) or not encoded:
            raise ValueError(f"CCCP layer {raw_layer} has no tier string")
        invalid = sorted(set(encoded) - set(_TIER_CHARACTER))
        if invalid:
            raise ValueError(
                f"CCCP layer {raw_layer} contains unsupported tier characters: "
                f"{invalid}"
            )
        if width is None:
            width = len(encoded)
        elif len(encoded) != width:
            raise ValueError(
                f"CCCP layer {raw_layer} has {len(encoded)} experts; "
                f"expected {width}"
            )
        result[int(raw_layer)] = tuple(
            _TIER_CHARACTER[character] for character in encoded
        )
    return result


def cccp_expert_precision(
    tier: str,
    *,
    artifact: str,
    iterations: int = 12,
    restarts: int = 2,
    sample_points: int = 100_000,
) -> ExpertPrecision:
    """Build a serializable CCCP cohort precision descriptor."""

    spec = _spec_for_tier(tier)
    return ExpertPrecision(
        family=spec.label,
        artifact=artifact,
        options=(
            ("iterations", int(iterations)),
            ("restarts", int(restarts)),
            ("sample_points", int(sample_points)),
        ),
    )


def _spec_for_tier(tier: str) -> CccpPqSpec:
    try:
        return TPQ_PQ_SPECS[f"TPQ-{str(tier).upper()}"]
    except KeyError as exc:
        raise ValueError(f"unsupported CCCP tier: {tier!r}") from exc


def build_cccp_expert_selection(
    *,
    name: str,
    group: str,
    allocation: CccpTierAllocation,
    rows_per_expert: int,
    columns: int,
    artifacts: Mapping[str, str],
) -> ExpertTensorSelection:
    """Create one MFQ calibration selection with exact index/codebook bits."""

    if rows_per_expert <= 0 or columns <= 0:
        raise ValueError("CCCP expert matrix dimensions must be positive")
    missing = sorted(set(allocation.tiers) - set(artifacts))
    if missing:
        raise ValueError(f"CCCP artifacts are missing tiers: {missing}")
    first_by_tier: dict[str, int] = {}
    for expert, tier in enumerate(allocation.tiers):
        first_by_tier.setdefault(tier, expert)
    selections: list[ExpertSelection] = []
    for expert, tier in enumerate(allocation.tiers):
        spec = _spec_for_tier(tier)
        if columns % spec.vector_size:
            raise ValueError(
                f"CCCP-{tier} cannot encode matrix width {columns}"
            )
        storage_bits = (
            rows_per_expert
            * (columns // spec.vector_size)
            * spec.index_bits
        )
        if first_by_tier[tier] == expert:
            storage_bits += (
                spec.codebook_entries * spec.vector_size * 32
            )
        precision = cccp_expert_precision(
            tier,
            artifact=str(artifacts[tier]),
        )
        selections.append(
            ExpertSelection(
                expert_id=expert,
                spec=None,
                precision=precision,
                storage_bits=int(storage_bits),
                train_loss=0.0,
                validation_loss=0.0,
            )
        )
    return ExpertTensorSelection(
        name=name,
        group=group,
        n_experts=len(allocation.tiers),
        rows_per_expert=rows_per_expert,
        columns=columns,
        selections=tuple(selections),
    )


__all__ = [
    "CccpTierAllocation",
    "allocate_cccp_tiers",
    "build_cccp_expert_selection",
    "cccp_expert_precision",
    "load_cccp_score_profile",
    "load_cccp_tier_profile",
]


# Canonical TPQ API aliases.
TpqTierAllocation = CccpTierAllocation
allocate_tpq_tiers = allocate_cccp_tiers
build_tpq_expert_selection = build_cccp_expert_selection
tpq_expert_precision = cccp_expert_precision
load_tpq_score_profile = load_cccp_score_profile
load_tpq_tier_profile = load_cccp_tier_profile

__all__ += [
    "TpqTierAllocation",
    "allocate_tpq_tiers",
    "build_tpq_expert_selection",
    "load_tpq_score_profile",
    "load_tpq_tier_profile",
    "tpq_expert_precision",
]
