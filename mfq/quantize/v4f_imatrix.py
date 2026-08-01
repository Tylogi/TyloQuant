"""DeepSeek-V4-Flash routed-expert imatrix bindings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mfq.quantize.imatrix import ImportanceMatrix, load_importance_matrix


_LAYERS = 43
_EXPERTS = 256
_WIDTHS = {"gate_up": 4096, "down": 2048}


@dataclass(frozen=True)
class V4FExpertImportance:
    layer: int
    projection: str
    entry_names: tuple[str, ...]
    values: np.ndarray
    counts: np.ndarray

    def count_weighted(self, expert_ids: np.ndarray | None = None) -> np.ndarray:
        values = self.values if expert_ids is None else self.values[expert_ids]
        counts = self.counts if expert_ids is None else self.counts[expert_ids]
        objective = values.astype(np.float32, copy=True)
        objective *= counts.astype(np.float32, copy=False)[:, None]
        mean = float(objective.mean(dtype=np.float64))
        if not np.isfinite(mean) or mean <= 0:
            raise ValueError("V4F count-weighted imatrix has no positive mass")
        objective *= 1.0 / mean
        return np.ascontiguousarray(objective)


class V4FImportanceMatrix:
    def __init__(self, matrix: ImportanceMatrix):
        self.matrix = matrix
        self._validate_complete_routed_set()

    @classmethod
    def load(cls, path: str | Path) -> "V4FImportanceMatrix":
        return cls(load_importance_matrix(path))

    def expert(self, layer: int, projection: str) -> V4FExpertImportance:
        if not 0 <= layer < _LAYERS:
            raise ValueError("V4F layer must be in [0,42]")
        if projection not in _WIDTHS:
            raise ValueError(f"unsupported V4F expert projection: {projection}")
        if projection == "gate_up":
            names = (
                f"blk.{layer}.ffn_gate_exps.weight",
                f"blk.{layer}.ffn_up_exps.weight",
            )
            gate = self.matrix.entries[names[0]]
            up = self.matrix.entries[names[1]]
            if not np.array_equal(gate.counts, up.counts):
                raise ValueError(f"V4F gate/up imatrix counts differ at layer {layer}")
            if not np.array_equal(gate.values, up.values):
                delta = float(np.max(np.abs(gate.values - up.values)))
                raise ValueError(
                    f"V4F gate/up imatrix values differ at layer {layer}: {delta}"
                )
            entry = gate
        else:
            names = (f"blk.{layer}.ffn_down_exps.weight",)
            entry = self.matrix.entries[names[0]]
        expected = (_EXPERTS, _WIDTHS[projection])
        if entry.values.shape != expected or entry.counts.shape != (_EXPERTS,):
            raise ValueError(
                f"V4F imatrix shape mismatch for {names[0]}: "
                f"{entry.values.shape}, {entry.counts.shape}"
            )
        return V4FExpertImportance(
            layer=layer,
            projection=projection,
            entry_names=names,
            values=np.ascontiguousarray(entry.values, dtype=np.float32),
            counts=np.ascontiguousarray(entry.counts, dtype=np.int64),
        )

    def _validate_complete_routed_set(self) -> None:
        for layer in range(_LAYERS):
            self.expert(layer, "gate_up")
            self.expert(layer, "down")


__all__ = [
    "V4FExpertImportance",
    "V4FImportanceMatrix",
]
