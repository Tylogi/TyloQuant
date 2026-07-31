"""MoEQuant affinity-guided diagonal calibration statistics.

MoEQuant's AGQ objective weights each routed expert sample by the router
coefficient ``c``.  For a linear projection this changes the diagonal input
second moment from ``sum(x**2)`` to ``sum(c * x**2)``.  The accumulator below
keeps both objectives so an experiment can change only the calibration metric
while holding samples, weights, and quantization format fixed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

Objective = Literal["count", "affinity"]
_FORMAT = "mfq.moequant-agq-diagonal.v1"


def _normalized(values: np.ndarray, label: str) -> np.ndarray:
    objective = np.asarray(values, dtype=np.float32)
    if objective.ndim != 2:
        raise ValueError(f"{label} objective must be rank 2")
    if not np.isfinite(objective).all() or np.any(objective < 0):
        raise ValueError(f"{label} objective must be finite and non-negative")
    mean = float(objective.mean(dtype=np.float64))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError(f"{label} objective has no positive mass")
    return np.ascontiguousarray(objective / np.float32(mean))


def diagonal_second_moment(
    inputs: np.ndarray,
    affinities: np.ndarray | None = None,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Return ``sum(x**2)`` or MoEQuant's ``sum(c * x**2)`` diagonal.

    ``inputs`` has shape ``[samples, width]``.  ``affinities`` is the selected
    router coefficient for every sample.  Normalization is a scalar-only
    rescaling and therefore does not change a weighted least-squares optimum.
    """

    values = np.asarray(inputs, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("inputs must be a non-empty [samples,width] matrix")
    if not np.isfinite(values).all():
        raise ValueError("inputs must be finite")
    squared = values.astype(np.float64) ** 2
    label = "count"
    if affinities is not None:
        weights = np.asarray(affinities, dtype=np.float64).reshape(-1)
        if weights.shape != (values.shape[0],):
            raise ValueError("affinities must have one value per input sample")
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("affinities must be finite and non-negative")
        squared *= weights[:, None]
        label = "affinity"
    result = squared.sum(axis=0, dtype=np.float64).astype(np.float32)[None, :]
    return _normalized(result, label)[0] if normalize else result[0]


@dataclass
class ExpertAffinityAccumulator:
    """Streaming count- and affinity-weighted moments for one MoE projection."""

    route_counts: np.ndarray
    affinity_sums: np.ndarray
    input_sum2: np.ndarray
    affinity_input_sum2: np.ndarray

    @classmethod
    def create(cls, experts: int, width: int) -> ExpertAffinityAccumulator:
        if experts <= 0 or width <= 0:
            raise ValueError("experts and width must be positive")
        return cls(
            route_counts=np.zeros(experts, dtype=np.int64),
            affinity_sums=np.zeros(experts, dtype=np.float64),
            input_sum2=np.zeros((experts, width), dtype=np.float64),
            affinity_input_sum2=np.zeros((experts, width), dtype=np.float64),
        )

    @property
    def experts(self) -> int:
        return int(self.route_counts.shape[0])

    @property
    def width(self) -> int:
        return int(self.input_sum2.shape[1])

    def __post_init__(self) -> None:
        self.route_counts = np.ascontiguousarray(self.route_counts, dtype=np.int64)
        self.affinity_sums = np.ascontiguousarray(self.affinity_sums, dtype=np.float64)
        self.input_sum2 = np.ascontiguousarray(self.input_sum2, dtype=np.float64)
        self.affinity_input_sum2 = np.ascontiguousarray(
            self.affinity_input_sum2, dtype=np.float64
        )
        experts = int(self.route_counts.size)
        if self.route_counts.shape != (experts,):
            raise ValueError("route_counts must be one-dimensional")
        if self.affinity_sums.shape != (experts,):
            raise ValueError("affinity_sums shape mismatch")
        if self.input_sum2.ndim != 2 or self.input_sum2.shape[0] != experts:
            raise ValueError("input_sum2 shape mismatch")
        if self.affinity_input_sum2.shape != self.input_sum2.shape:
            raise ValueError("affinity_input_sum2 shape mismatch")
        if np.any(self.route_counts < 0):
            raise ValueError("route counts must be non-negative")
        for label, values in (
            ("affinity sums", self.affinity_sums),
            ("input sums", self.input_sum2),
            ("affinity input sums", self.affinity_input_sum2),
        ):
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{label} must be finite and non-negative")

    def update(
        self,
        inputs: np.ndarray,
        expert_ids: np.ndarray,
        affinities: np.ndarray,
    ) -> None:
        values = np.asarray(inputs, dtype=np.float32)
        ids = np.asarray(expert_ids, dtype=np.int64).reshape(-1)
        weights = np.asarray(affinities, dtype=np.float64).reshape(-1)
        if values.ndim != 2 or values.shape[1] != self.width:
            raise ValueError(f"inputs must have shape [samples,{self.width}]")
        if ids.shape != (values.shape[0],) or weights.shape != ids.shape:
            raise ValueError("expert_ids and affinities must match input samples")
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= self.experts):
            raise IndexError("expert id is outside the accumulator")
        if not np.isfinite(values).all():
            raise ValueError("inputs must be finite")
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("affinities must be finite and non-negative")

        squared = values.astype(np.float64) ** 2
        for expert in np.unique(ids):
            selected = ids == expert
            local_weights = weights[selected]
            self.route_counts[expert] += int(selected.sum())
            self.affinity_sums[expert] += local_weights.sum(dtype=np.float64)
            self.input_sum2[expert] += squared[selected].sum(axis=0, dtype=np.float64)
            self.affinity_input_sum2[expert] += (
                squared[selected] * local_weights[:, None]
            ).sum(axis=0, dtype=np.float64)

    def merge(self, other: ExpertAffinityAccumulator) -> None:
        if other.input_sum2.shape != self.input_sum2.shape:
            raise ValueError("cannot merge accumulators with different shapes")
        self.route_counts += other.route_counts
        self.affinity_sums += other.affinity_sums
        self.input_sum2 += other.input_sum2
        self.affinity_input_sum2 += other.affinity_input_sum2

    def objective(
        self,
        kind: Objective,
        expert_ids: np.ndarray | None = None,
        *,
        normalize: bool = True,
    ) -> np.ndarray:
        if kind == "count":
            values = self.input_sum2
        elif kind == "affinity":
            values = self.affinity_input_sum2
        else:
            raise ValueError(f"unsupported MoEQuant objective: {kind}")
        selected = values if expert_ids is None else values[np.asarray(expert_ids, dtype=np.int64)]
        result = np.ascontiguousarray(selected, dtype=np.float32)
        return _normalized(result, kind) if normalize else result

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                format=np.asarray(_FORMAT),
                route_counts=self.route_counts,
                affinity_sums=self.affinity_sums,
                input_sum2=self.input_sum2,
                affinity_input_sum2=self.affinity_input_sum2,
                metadata=np.asarray(json.dumps(metadata or {}, sort_keys=True)),
            )
        os.replace(temporary, output)

    @classmethod
    def load(cls, path: str | Path) -> tuple[ExpertAffinityAccumulator, dict[str, Any]]:
        with np.load(Path(path), allow_pickle=False) as payload:
            if str(payload["format"].item()) != _FORMAT:
                raise ValueError("unsupported MoEQuant AGQ artifact format")
            result = cls(
                route_counts=payload["route_counts"],
                affinity_sums=payload["affinity_sums"],
                input_sum2=payload["input_sum2"],
                affinity_input_sum2=payload["affinity_input_sum2"],
            )
            metadata = json.loads(str(payload["metadata"].item()))
        if not isinstance(metadata, dict):
            raise ValueError("MoEQuant AGQ artifact metadata must be an object")
        return result, metadata


__all__ = [
    "ExpertAffinityAccumulator",
    "Objective",
    "diagonal_second_moment",
]
