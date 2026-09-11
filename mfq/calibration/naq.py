"""Architecture-neutral NAQ-imatrix evidence and registry types.

The activation collector records two independent factors for a weight matrix:
input-channel second moments and output-neuron sensitivity.  This module turns
that persisted evidence into immutable, canonical tensor-local maps.  It does
not know model architectures, tensor aliases, quantization formats, or writer
layouts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Any

import numpy as np

from mfq.quantize.imatrix import ImportanceEntry, ImportanceMatrix

TensorKey = str


def _readonly(array: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _row_ids(rows: slice | Iterable[int] | np.ndarray, row_count: int) -> np.ndarray:
    if isinstance(rows, slice):
        start, stop, step = rows.indices(row_count)
        result = np.arange(start, stop, step, dtype=np.int64)
    else:
        result = np.asarray(rows, dtype=np.int64).reshape(-1)
    if result.size and (int(result.min()) < 0 or int(result.max()) >= row_count):
        raise IndexError(f"NAQ row selection is outside {row_count} rows")
    return result


@dataclass(frozen=True)
class TensorRateBudget:
    """Exact physical budget for one canonical tensor."""

    target_bits: int
    value_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target_bits, Integral)
            or isinstance(self.target_bits, bool)
            or not isinstance(self.value_count, Integral)
            or isinstance(self.value_count, bool)
            or self.target_bits <= 0
            or self.value_count <= 0
        ):
            raise ValueError("tensor rate budgets must contain positive integers")

    @property
    def bpw(self) -> float:
        return self.target_bits / self.value_count


@dataclass(frozen=True)
class NaqIMap:
    """Factorized within-tensor importance used by NAQ-aware solvers.

    ``input_moments`` may contain one vector for a dense tensor, one per expert,
    or one per neuron. ``row_to_input`` maps every logical output neuron to the
    corresponding vector without expanding a full ``[out, in]`` matrix.
    """

    tensor_shape: tuple[int, int]
    input_moments: np.ndarray
    neuron_factors: np.ndarray
    row_to_input: np.ndarray
    objective_fingerprint: str = "naq-imatrix-v1"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rows, columns = map(int, self.tensor_shape)
        if rows <= 0 or columns <= 0:
            raise ValueError("NAQ tensor shape must be a non-empty matrix")
        inputs = np.asarray(self.input_moments, dtype=np.float32)
        neurons = np.asarray(self.neuron_factors, dtype=np.float32).reshape(-1)
        indices = np.asarray(self.row_to_input, dtype=np.int64).reshape(-1)
        if inputs.ndim != 2 or inputs.shape[1] != columns or not inputs.shape[0]:
            raise ValueError(
                "NAQ input moments must have shape [factor_count, input_width]"
            )
        if neurons.shape != (rows,) or indices.shape != (rows,):
            raise ValueError("NAQ neuron factors and row mapping must match output rows")
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= inputs.shape[0]):
            raise ValueError("NAQ row mapping references a missing input-moment vector")
        if (
            not np.isfinite(inputs).all()
            or not np.isfinite(neurons).all()
            or np.any(inputs < 0)
            or np.any(neurons < 0)
            or not np.any(inputs > 0)
            or not np.any(neurons > 0)
        ):
            raise ValueError("NAQ importance must be finite, non-negative, and non-zero")
        if not self.objective_fingerprint:
            raise ValueError("NAQ objective fingerprint cannot be empty")
        object.__setattr__(self, "input_moments", _readonly(inputs, np.float32))
        object.__setattr__(self, "neuron_factors", _readonly(neurons, np.float32))
        object.__setattr__(self, "row_to_input", _readonly(indices, np.int64))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def shape(self) -> tuple[int, int]:
        return self.tensor_shape

    def channel_importance(
        self, rows: slice | Iterable[int] | np.ndarray
    ) -> np.ndarray:
        ids = _row_ids(rows, self.tensor_shape[0])
        return np.ascontiguousarray(
            self.input_moments[self.row_to_input[ids]], dtype=np.float32
        )

    def neuron_importance(
        self, rows: slice | Iterable[int] | np.ndarray
    ) -> np.ndarray:
        ids = _row_ids(rows, self.tensor_shape[0])
        return np.ascontiguousarray(self.neuron_factors[ids], dtype=np.float32)

    def element_importance(
        self, rows: slice | Iterable[int] | np.ndarray
    ) -> np.ndarray:
        ids = _row_ids(rows, self.tensor_shape[0])
        return np.ascontiguousarray(
            self.input_moments[self.row_to_input[ids]]
            * self.neuron_factors[ids, None],
            dtype=np.float32,
        )

    def weighted_row_loss(
        self,
        error: np.ndarray,
        rows: slice | Iterable[int] | np.ndarray,
    ) -> np.ndarray:
        ids = _row_ids(rows, self.tensor_shape[0])
        values = np.asarray(error, dtype=np.float64)
        expected = (ids.size, self.tensor_shape[1])
        if values.shape != expected:
            raise ValueError(f"NAQ error shape {values.shape} != {expected}")
        importance = self.element_importance(ids).astype(np.float64)
        return np.ascontiguousarray((importance * values * values).sum(axis=1))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"mfq.naq.imap.v1\0")
        digest.update(np.asarray(self.tensor_shape, dtype=np.int64).tobytes())
        digest.update(self.input_moments.tobytes())
        digest.update(self.neuron_factors.tobytes())
        digest.update(self.row_to_input.tobytes())
        digest.update(self.objective_fingerprint.encode("utf-8"))
        digest.update(
            json.dumps(
                dict(self.provenance), sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
        )
        return digest.hexdigest()

    @classmethod
    def from_entry(
        cls,
        entry: ImportanceEntry,
        *,
        original_shape: tuple[int, ...],
        storage_shape: tuple[int, int],
        objective_fingerprint: str = "naq-imatrix-v1",
        provenance: Mapping[str, Any] | None = None,
    ) -> NaqIMap:
        rows, columns = map(int, storage_shape)
        values = np.asarray(entry.values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != columns:
            raise ValueError(
                f"NAQ input-moment shape {values.shape} is incompatible with {storage_shape}"
            )
        if values.shape[0] == 1:
            row_to_input = np.zeros(rows, dtype=np.int64)
        elif values.shape[0] == rows:
            row_to_input = np.arange(rows, dtype=np.int64)
        elif len(original_shape) == 3:
            experts, rows_per_expert, width = map(int, original_shape)
            if (
                values.shape[0] != experts
                or rows != experts * rows_per_expert
                or width != columns
            ):
                raise ValueError(
                    "NAQ expert factors do not match the original and storage shapes"
                )
            row_to_input = np.repeat(np.arange(experts, dtype=np.int64), rows_per_expert)
        else:
            raise ValueError(
                f"NAQ entry has {values.shape[0]} input vectors for {rows} rows"
            )
        neuron_importance = (
            np.ones(rows, dtype=np.float32)
            if entry.row_importance is None
            else np.asarray(entry.row_importance, dtype=np.float32).reshape(-1)
        )
        return cls(
            tensor_shape=(rows, columns),
            input_moments=values,
            neuron_factors=neuron_importance,
            row_to_input=row_to_input,
            objective_fingerprint=objective_fingerprint,
            provenance={} if provenance is None else provenance,
        )


@dataclass(frozen=True)
class NaqTensorBinding:
    """Canonical tensor geometry plus legacy artifact lookup names."""

    tensor_key: TensorKey
    entry_names: tuple[str, ...]
    original_shape: tuple[int, ...]
    storage_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if not self.tensor_key or not self.entry_names:
            raise ValueError("NAQ tensor bindings require a key and at least one entry name")
        if len(self.storage_shape) != 2 or any(int(value) <= 0 for value in self.storage_shape):
            raise ValueError("NAQ storage shape must be a non-empty matrix")


@dataclass(frozen=True)
class NaqIBook:
    """Canonical model-wide registry of NAQ IMaps and exact tensor budgets."""

    imaps: Mapping[TensorKey, NaqIMap]
    budgets: Mapping[TensorKey, TensorRateBudget]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        imaps = dict(self.imaps)
        budgets = dict(self.budgets)
        if not imaps or set(imaps) != set(budgets):
            raise ValueError("NAQ IBook IMaps and budgets must cover the same tensors")
        if any(not key for key in imaps):
            raise ValueError("NAQ IBook tensor keys must be canonical non-empty names")
        for key, imap in imaps.items():
            if imap.shape[0] * imap.shape[1] != budgets[key].value_count:
                raise ValueError(f"NAQ budget value count does not match {key}")
        object.__setattr__(self, "imaps", MappingProxyType(imaps))
        object.__setattr__(self, "budgets", MappingProxyType(budgets))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def imap(self, tensor_key: TensorKey) -> NaqIMap:
        return self.imaps[tensor_key]

    def budget(self, tensor_key: TensorKey) -> TensorRateBudget:
        return self.budgets[tensor_key]

    def validate_inventory(self, tensor_keys: Iterable[TensorKey]) -> None:
        expected = set(tensor_keys)
        actual = set(self.imaps)
        if expected != actual:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"NAQ IBook inventory mismatch: missing={missing}, extra={extra}")

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"mfq.naq.ibook.v1\0")
        for key in sorted(self.imaps):
            budget = self.budgets[key]
            digest.update(key.encode("utf-8"))
            digest.update(self.imaps[key].fingerprint().encode("ascii"))
            digest.update(np.asarray([budget.target_bits, budget.value_count], dtype=np.int64).tobytes())
        digest.update(
            json.dumps(
                dict(self.provenance), sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
        )
        return digest.hexdigest()


class NaqImatrixCalibrator:
    """Compile persisted NAQ statistics into a canonical immutable IBook."""

    def __init__(self, objective_fingerprint: str = "naq-imatrix-v1") -> None:
        if not objective_fingerprint:
            raise ValueError("NAQ calibrator objective fingerprint cannot be empty")
        self.objective_fingerprint = objective_fingerprint

    def calibrate(
        self,
        imatrix: ImportanceMatrix,
        bindings: Iterable[NaqTensorBinding],
        budgets: Mapping[TensorKey, TensorRateBudget],
    ) -> NaqIBook:
        maps: dict[TensorKey, NaqIMap] = {}
        for binding in bindings:
            if binding.tensor_key in maps:
                raise ValueError(f"duplicate canonical NAQ tensor: {binding.tensor_key}")
            match = imatrix.find(binding.entry_names)
            if match is None:
                raise KeyError(
                    f"NAQ artifact has no entry for {binding.tensor_key}: {binding.entry_names}"
                )
            entry_name, entry = match
            maps[binding.tensor_key] = NaqIMap.from_entry(
                entry,
                original_shape=binding.original_shape,
                storage_shape=binding.storage_shape,
                objective_fingerprint=self.objective_fingerprint,
                provenance={"entry_name": entry_name},
            )
        provenance = {
            "format": imatrix.metadata.get("format", "mfq.imatrix.v1"),
            "datasets": tuple(imatrix.datasets),
            "chunk_count": int(imatrix.chunk_count),
            "chunk_size": int(imatrix.chunk_size),
            "objective": self.objective_fingerprint,
        }
        return NaqIBook(maps, budgets, provenance)


__all__ = [
    "NaqIBook",
    "NaqIMap",
    "NaqImatrixCalibrator",
    "NaqTensorBinding",
    "TensorKey",
    "TensorRateBudget",
]
