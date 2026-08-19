"""Evaluate NINT profiles against collected function-level statistics."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mfq.calibration.allocator import GroupCandidate, allocate
from mfq.calibration.artifact import (
    CalibrationScheme,
    ExpertPrecision,
    TensorSelection,
    nint_expert_precision,
    precision_document,
    precision_from_document,
    save_scheme,
)
from mfq.calibration.dense_precision import (
    DensePrecision,
    dense_candidate_reconstruction,
    dense_candidate_storage_bits,
    dense_precision_cache_path,
    load_dense_candidate,
    normalize_dense_precision,
    quantize_dense_precision,
    save_dense_candidate,
    slice_dense_tensor_rows,
)
from mfq.calibration.qwen35 import HfSafetensorIndex
from mfq.calibration.statistics import CalibrationStatistics, TensorStatistics
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import NintTensor
from mfq.quantize.nint_quant import quantize as quantize_nint_cpu
from mfq.quantize.nint_quant_torch import quantize_axis0 as quantize_nint_cuda

NINT_CALIBRATION_PROFILES: dict[str, NintSpec] = {
    "NINT4": NintSpec(4, 24, 6),
    "NINT5": NintSpec(5, 28, 7),
    "NINT6": NintSpec(6, 24, 7),
    "NINT8": NintSpec(8, 48, 7),
}
NINT_EXPERT_PROFILES: dict[str, NintSpec] = {
    "NINT2": NintSpec(2, 16, 5),
    "NINT3": NintSpec(3, 24, 5),
    **NINT_CALIBRATION_PROFILES,
}
DENSE_PROFILE_FAMILIES = (
    "NPQ0-S",
    "NPQ0-L",
    "NVQ2",
    "NVQ2J",
    "NVQ2J-L",
    "NVQ2J-XL",
    "NVQ3",
    "NVQ3J",
    "NVQ3J-512",
    "NVQ3J-L",
    "NINT2",
    "NINT3",
    "NINT4",
    "NINT5",
    "NINT6",
    "NINT8",
    "NINT8-0",
)


@dataclass(frozen=True)
class TensorCandidateEvaluation:
    name: str
    group: str
    profile: str
    spec: NintSpec | None
    rows: int
    columns: int
    storage_bits: int
    train_loss: float
    validation_loss: float
    train_nmse_percent: float
    validation_nmse_percent: float
    train_row_loss: np.ndarray
    validation_row_loss: np.ndarray
    precision: ExpertPrecision | None = None

    @property
    def descriptor(self) -> ExpertPrecision:
        if self.precision is None:
            if self.spec is None:
                raise ValueError(f"candidate {self.name}/{self.profile} has no precision")
            return nint_expert_precision(self.spec)
        if self.spec is not None and self.precision.nint_spec != self.spec:
            raise ValueError(f"candidate {self.name}/{self.profile} descriptors disagree")
        return self.precision


@dataclass(frozen=True)
class LayerStrategy:
    layer: int
    name: str
    specs: dict[str, DensePrecision]
    profiles: dict[str, str]
    storage_bits: int
    train_loss: float
    validation_loss: float


def nint_storage_bits(rows: int, columns: int, spec: NintSpec) -> int:
    if rows <= 0 or columns <= 0:
        raise ValueError("NINT storage shape must be positive")
    groups = (columns + spec.groupsize - 1) // spec.groupsize
    return int(
        rows * 32 + rows * groups * 2 * spec.sub_bits + rows * groups * spec.groupsize * spec.bits
    )


def nint_row_storage_bits(columns: int, spec: NintSpec) -> int:
    return nint_storage_bits(1, columns, spec)


def _packed_row_objective(
    reference: torch.Tensor,
    encoded,
    input_second_moment: np.ndarray,
    row_fisher: np.ndarray,
    *,
    row_chunk: int,
) -> tuple[np.ndarray, float]:
    rows, columns = (int(reference.shape[0]), int(reference.shape[1]))
    if input_second_moment.shape != (columns,) or row_fisher.shape != (rows,):
        raise ValueError("candidate statistics do not match the weight shape")
    losses = np.empty(rows, dtype=np.float64)
    reference_energy = 0.0
    input_weight = np.asarray(input_second_moment, dtype=np.float64)
    fisher = np.asarray(row_fisher, dtype=np.float64)
    for start in range(0, rows, row_chunk):
        end = min(start + row_chunk, rows)
        weight = reference[start:end].float().cpu().numpy().astype(np.float64, copy=False)
        packed_rows = slice_dense_tensor_rows(encoded, start, end)
        restored = dense_candidate_reconstruction(packed_rows).astype(
            np.float64,
            copy=False,
        )
        error = weight - restored
        local_fisher = fisher[start:end]
        losses[start:end] = ((error * error) * input_weight[None]).sum(axis=1) * local_fisher
        reference_energy += float(
            (((weight * weight) * input_weight[None]).sum(axis=1) * local_fisher).sum()
        )
    return np.ascontiguousarray(losses, dtype=np.float32), reference_energy


def _quantize(
    weight: torch.Tensor,
    spec: NintSpec,
    *,
    backend: str,
    device: str,
) -> NintTensor:
    if backend in {"cuda", "metal"}:
        if backend == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA candidate evaluation requested without CUDA")
        return quantize_nint_cuda(weight, spec, device=device)
    if backend == "cpu":
        return quantize_nint_cpu(weight.float().cpu().numpy(), spec, axis=0)
    raise ValueError("candidate quantization backend must be cuda or cpu")


def evaluate_tensor_candidate(
    weight: torch.Tensor,
    statistics: TensorStatistics,
    profile: str,
    spec: NintSpec,
    *,
    backend: str = "cuda",
    device: str = "cuda:0",
    row_chunk: int = 256,
    encoded: NintTensor | None = None,
) -> TensorCandidateEvaluation:
    target = statistics.target
    if tuple(weight.shape) != (target.rows, target.columns):
        raise ValueError(
            f"weight shape for {target.name} is {tuple(weight.shape)}, "
            f"expected {(target.rows, target.columns)}"
        )
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")
    if encoded is None:
        encoded = _quantize(weight, spec, backend=backend, device=device)
    elif (
        encoded.spec != spec
        or encoded.shape != (target.rows, target.columns)
        or encoded.axis != 0
        or encoded.neuron_len != target.columns
    ):
        raise ValueError(f"packed NINT candidate does not match {target.name} {profile}")
    train_rows, train_energy = _packed_row_objective(
        weight,
        encoded,
        statistics.train_input_second_moment,
        statistics.train_row_fisher,
        row_chunk=row_chunk,
    )
    validation_rows, validation_energy = _packed_row_objective(
        weight,
        encoded,
        statistics.validation_input_second_moment,
        statistics.validation_row_fisher,
        row_chunk=row_chunk,
    )
    train_loss = float(train_rows.astype(np.float64).sum())
    validation_loss = float(validation_rows.astype(np.float64).sum())
    return TensorCandidateEvaluation(
        name=target.name,
        group=target.group,
        profile=profile,
        spec=spec,
        rows=target.rows,
        columns=target.columns,
        storage_bits=nint_storage_bits(target.rows, target.columns, spec),
        train_loss=train_loss,
        validation_loss=validation_loss,
        train_nmse_percent=100.0 * train_loss / max(train_energy, np.finfo(np.float64).tiny),
        validation_nmse_percent=(
            100.0 * validation_loss / max(validation_energy, np.finfo(np.float64).tiny)
        ),
        train_row_loss=train_rows,
        validation_row_loss=validation_rows,
        precision=nint_expert_precision(spec),
    )


def evaluate_tensor_precision(
    weight: torch.Tensor,
    statistics: TensorStatistics,
    profile: str,
    precision: DensePrecision,
    *,
    backend: str = "cuda",
    device: str = "cuda:0",
    row_chunk: int = 256,
    encoded=None,
    artifact_root: str | Path | None = None,
    quantization_importance: np.ndarray | torch.Tensor | None = None,
) -> tuple[TensorCandidateEvaluation, object]:
    """Quantize and score one heterogeneous packed Dense candidate."""

    descriptor = normalize_dense_precision(precision)
    target = statistics.target
    if tuple(weight.shape) != (target.rows, target.columns):
        raise ValueError(f"weight shape does not match {target.name}")
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")
    if encoded is None:
        encoded = quantize_dense_precision(
            weight,
            descriptor,
            importance=(
                statistics.train_input_second_moment
                if quantization_importance is None
                else quantization_importance
            ),
            backend=backend,
            device=device,
            artifact_root=artifact_root,
        )
    encoded_shape = tuple(int(value) for value in getattr(encoded, "shape", ()))
    if encoded_shape != (target.rows, target.columns):
        raise ValueError(f"packed candidate reconstruction does not match {target.name}")
    train_rows, train_energy = _packed_row_objective(
        weight,
        encoded,
        statistics.train_input_second_moment,
        statistics.train_row_fisher,
        row_chunk=row_chunk,
    )
    validation_rows, validation_energy = _packed_row_objective(
        weight,
        encoded,
        statistics.validation_input_second_moment,
        statistics.validation_row_fisher,
        row_chunk=row_chunk,
    )
    train_loss = float(train_rows.astype(np.float64).sum())
    validation_loss = float(validation_rows.astype(np.float64).sum())
    value = TensorCandidateEvaluation(
        name=target.name,
        group=target.group,
        profile=profile,
        spec=descriptor.nint_spec,
        rows=target.rows,
        columns=target.columns,
        storage_bits=(
            nint_storage_bits(target.rows, target.columns, descriptor.nint_spec)
            if descriptor.nint_spec is not None
            else dense_candidate_storage_bits(encoded)
        ),
        train_loss=train_loss,
        validation_loss=validation_loss,
        train_nmse_percent=100.0 * train_loss / max(train_energy, np.finfo(np.float64).tiny),
        validation_nmse_percent=(
            100.0 * validation_loss / max(validation_energy, np.finfo(np.float64).tiny)
        ),
        train_row_loss=train_rows,
        validation_row_loss=validation_rows,
        precision=descriptor,
    )
    return value, encoded


def _identity_digest(
    statistics: CalibrationStatistics,
    model_root: Path,
    *,
    extra_identity: str = "",
) -> str:
    digest = hashlib.sha256()
    with statistics.path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    for filename in ("config.json", "model.safetensors.index.json"):
        path = model_root / filename
        if path.is_file():
            with path.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    digest.update(chunk)
    digest.update(extra_identity.encode("utf-8"))
    return digest.hexdigest()


def _cache_path(
    root: Path,
    name: str,
    profile: str,
    identity: str | None = None,
) -> Path:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:20]
    identity_suffix = "" if identity is None else f"-{identity[:16]}"
    return root / f"{digest}-{profile.lower()}{identity_suffix}.npz"


def _save_candidate_cache(
    path: Path,
    value: TensorCandidateEvaluation,
    identity: str,
) -> None:
    document = {
        "identity": identity,
        "name": value.name,
        "group": value.group,
        "profile": value.profile,
        "precision": precision_document(value.descriptor),
        **(
            {
                "spec": {
                    "bits": value.spec.bits,
                    "groupsize": value.spec.groupsize,
                    "sub_bits": value.spec.sub_bits,
                }
            }
            if value.spec is not None
            else {}
        ),
        "rows": value.rows,
        "columns": value.columns,
        "storage_bits": value.storage_bits,
        "train_loss": value.train_loss,
        "validation_loss": value.validation_loss,
        "train_nmse_percent": value.train_nmse_percent,
        "validation_nmse_percent": value.validation_nmse_percent,
    }
    metadata = np.frombuffer(json.dumps(document, separators=(",", ":")).encode(), dtype=np.uint8)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata=metadata,
            train_row_loss=value.train_row_loss,
            validation_row_loss=value.validation_row_loss,
        )
    os.replace(temporary, path)


def _load_candidate_cache(
    path: Path,
    identity: str,
) -> TensorCandidateEvaluation | None:
    loaded = _read_candidate_cache(path)
    if loaded is None:
        return None
    document, value = loaded
    if document.get("identity") != identity:
        return None
    return value


def _read_candidate_cache(
    path: Path,
) -> tuple[dict[str, Any], TensorCandidateEvaluation] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        document = json.loads(archive["metadata"].tobytes().decode())
        if "precision" in document:
            precision = precision_from_document(document["precision"])
            spec = precision.nint_spec
        else:
            spec = NintSpec(**document["spec"])
            precision = nint_expert_precision(spec)
        value = TensorCandidateEvaluation(
            name=str(document["name"]),
            group=str(document["group"]),
            profile=str(document["profile"]),
            spec=spec,
            rows=int(document["rows"]),
            columns=int(document["columns"]),
            storage_bits=int(document["storage_bits"]),
            train_loss=float(document["train_loss"]),
            validation_loss=float(document["validation_loss"]),
            train_nmse_percent=float(document["train_nmse_percent"]),
            validation_nmse_percent=float(document["validation_nmse_percent"]),
            train_row_loss=np.ascontiguousarray(archive["train_row_loss"], dtype=np.float32),
            validation_row_loss=np.ascontiguousarray(
                archive["validation_row_loss"], dtype=np.float32
            ),
            precision=precision,
        )
    return document, value


def _materialize_scheme_candidate_cache(
    path: Path,
    value: TensorCandidateEvaluation,
    identity: str,
) -> None:
    """Publish the stable cache entry consumed by a saved allocation scheme."""

    loaded = _read_candidate_cache(path)
    if loaded is None:
        _save_candidate_cache(path, value, identity)
        return
    document, _existing = loaded
    if document.get("identity") != identity:
        raise ValueError(
            "candidate cache contains a stable entry from another model or "
            f"statistics artifact: {path}"
        )


def load_scheme_candidate_evaluations(
    scheme: CalibrationScheme,
    cache_dir: str | Path,
    *,
    profiles: Mapping[str, DensePrecision] = NINT_CALIBRATION_PROFILES,
) -> tuple[dict[str, dict[str, TensorCandidateEvaluation]], dict[str, Any]]:
    """Load the exact legacy score set referenced by a scheme without rewriting it."""

    root = Path(cache_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"candidate cache does not exist: {root}")
    group_profiles: dict[str, dict[str, Mapping[str, Any]]] = {}
    for group, raw_options in scheme.candidate_table.items():
        options: dict[str, Mapping[str, Any]] = {}
        for raw in raw_options:
            profile = str(raw.get("profile", ""))
            if profile in options:
                raise ValueError(f"candidate table repeats {group}/{profile}")
            if "precisions" not in raw and profile not in profiles:
                raise ValueError(f"candidate table uses unknown profile {group}/{profile}")
            options[profile] = raw
        if not options:
            raise ValueError(f"candidate table group {group} has no profiles")
        group_profiles[group] = options

    evaluations: dict[str, dict[str, TensorCandidateEvaluation]] = {}
    identity_counts: dict[str, int] = {}
    for name, selection in sorted(scheme.selections.items()):
        options = group_profiles.get(selection.group)
        if options is None:
            continue
        values: dict[str, TensorCandidateEvaluation] = {}
        for profile in sorted(options):
            path = _cache_path(root, name, profile)
            loaded = _read_candidate_cache(path)
            if loaded is None:
                raise FileNotFoundError(
                    f"scheme candidate cache is missing {name}/{profile}: {path}"
                )
            document, value = loaded
            identity = str(document.get("identity", ""))
            identity_counts[identity] = identity_counts.get(identity, 0) + 1
            raw_precisions = options[profile].get("precisions")
            expected_precision = (
                precision_from_document(raw_precisions[name])
                if isinstance(raw_precisions, Mapping)
                else nint_expert_precision(profiles[profile])
            )
            if (
                value.name != name
                or value.group != selection.group
                or value.profile != profile
                or value.descriptor != expected_precision
                or value.rows != selection.rows
                or value.columns != selection.columns
            ):
                raise ValueError(f"scheme candidate cache metadata mismatch: {path}")
            values[profile] = value
        evaluations[name] = values

    for group, options in group_profiles.items():
        names = sorted(
            name for name, selection in scheme.selections.items() if selection.group == group
        )
        if not names:
            raise ValueError(f"candidate table group {group} has no selected tensors")
        for profile, raw in options.items():
            values = [evaluations[name][profile] for name in names]
            actual_storage = sum(value.storage_bits for value in values)
            expected_storage = int(raw["storage_bits"])
            if actual_storage != expected_storage:
                raise ValueError(
                    f"scheme candidate cache storage mismatch for {group}/{profile}: "
                    f"{actual_storage} != {expected_storage}"
                )
            for field in ("train_loss", "validation_loss"):
                actual = math.fsum(float(getattr(value, field)) for value in values)
                expected = float(raw[field])
                if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
                    raise ValueError(
                        f"scheme candidate cache {field} mismatch for {group}/{profile}: "
                        f"{actual} != {expected}"
                    )

    audit = {
        "path": str(root),
        "files": sum(len(values) for values in evaluations.values()),
        "tensors": len(evaluations),
        "identities": dict(sorted(identity_counts.items())),
        "validation": "exact_metadata_storage_and_group_loss_match",
        "write_mode": "read_only",
    }
    return evaluations, audit


def _packed_candidate_index(root: Path | None) -> dict[tuple[str, int, int, int], Path]:
    if root is None:
        return {}
    if not root.is_dir():
        raise FileNotFoundError(f"packed calibration candidate cache does not exist: {root}")
    result: dict[tuple[str, int, int, int], Path] = {}
    for path in sorted(root.glob("*.nint-exec.npz")):
        with np.load(path, allow_pickle=False) as archive:
            document = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        key = (
            str(document["name"]),
            int(document["bits"]),
            int(document["groupsize"]),
            int(document["sub_bits"]),
        )
        if key in result:
            raise ValueError(f"duplicate packed calibration candidate for {key}")
        result[key] = path
    return result


def _unpack_packed_q(q_packed: np.ndarray, bits: int, groupsize: int) -> np.ndarray:
    values = np.asarray(q_packed, dtype=np.uint8)
    if values.ndim != 3:
        raise ValueError("packed NINT values must have [out, groups, bytes] shape")
    expected_bytes = (groupsize * bits + 7) // 8
    if values.shape[-1] != expected_bytes:
        raise ValueError(
            f"packed NINT group has {values.shape[-1]} bytes; expected {expected_bytes}"
        )
    unpacked = np.unpackbits(values, axis=-1, count=groupsize * bits, bitorder="little")
    unpacked = unpacked.reshape(*values.shape[:-1], groupsize, bits)
    powers = np.asarray(1 << np.arange(bits), dtype=np.uint16)
    return np.ascontiguousarray((unpacked * powers).sum(axis=-1), dtype=np.uint8)


def _load_packed_candidate(
    path: Path,
    statistics: TensorStatistics,
    profile: str,
    spec: NintSpec,
) -> NintTensor:
    target = statistics.target
    with np.load(path, allow_pickle=False) as archive:
        document = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        expected = {
            "name": target.name,
            "bits": spec.bits,
            "groupsize": spec.groupsize,
            "sub_bits": spec.sub_bits,
            "shape": [target.rows, target.columns],
            "axis": 0,
            "neuron_len": target.columns,
        }
        if document != expected:
            raise ValueError(f"packed NINT candidate metadata does not match {profile}: {path}")
        q = _unpack_packed_q(archive["q_packed"], spec.bits, spec.groupsize)
        sub_scale = np.ascontiguousarray(archive["sub_scale"])
        sub_min = np.ascontiguousarray(archive["sub_min"])
        neuron_scale = np.ascontiguousarray(archive["neuron_scale"], dtype=np.float32)
        neuron_min = np.ascontiguousarray(archive["neuron_min"], dtype=np.float32)
    return NintTensor(
        spec=spec,
        shape=(target.rows, target.columns),
        axis=0,
        q=q,
        neuron_scale=neuron_scale,
        neuron_min=neuron_min,
        sub_scale=sub_scale,
        sub_min=sub_min,
        neuron_len=target.columns,
    )


def evaluate_candidates(
    model_path: str | Path,
    statistics: CalibrationStatistics,
    *,
    profiles: Mapping[str, DensePrecision] = NINT_CALIBRATION_PROFILES,
    cache_dir: str | Path,
    backend: str = "cuda",
    device: str = "cuda:0",
    row_chunk: int = 256,
    packed_cache_dir: str | Path | None = None,
    target_names: set[str] | frozenset[str] | None = None,
    quantization_importance: Mapping[str, np.ndarray] | None = None,
    extra_identity: str = "",
) -> dict[str, dict[str, TensorCandidateEvaluation]]:
    root = Path(model_path).resolve()
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    identity = _identity_digest(
        statistics,
        root,
        extra_identity=extra_identity,
    )
    source = HfSafetensorIndex(root)
    if packed_cache_dir is not None:
        Path(packed_cache_dir).resolve().mkdir(parents=True, exist_ok=True)
    packed_index = _packed_candidate_index(
        None if packed_cache_dir is None else Path(packed_cache_dir).resolve()
    )
    if packed_index:
        print(
            json.dumps(
                {
                    "event": "packed_candidate_cache_ready",
                    "path": str(Path(packed_cache_dir).resolve()),
                    "files": len(packed_index),
                    "tensors": len({key[0] for key in packed_index}),
                }
            ),
            flush=True,
        )
    result: dict[str, dict[str, TensorCandidateEvaluation]] = {}
    entries = {
        name: stats
        for name, stats in statistics.entries.items()
        if target_names is None or name in target_names
    }
    if target_names is not None:
        missing = sorted(set(target_names) - set(entries))
        if missing:
            raise ValueError(f"candidate target filter contains unknown tensors: {missing[:8]}")
    total = len(entries) * len(profiles)
    done = 0
    started = time.time()

    for name, stats in sorted(entries.items()):
        target = stats.target
        weight: torch.Tensor | None = None
        result[name] = {}
        for profile, raw_precision in profiles.items():
            precision = normalize_dense_precision(raw_precision)
            done += 1
            path = _cache_path(cache, name, profile, identity)
            value = _load_candidate_cache(path, identity)
            legacy_path = _cache_path(cache, name, profile)
            if value is None:
                value = _load_candidate_cache(legacy_path, identity)
            cached = value is not None
            if value is not None and value.descriptor != precision:
                value = None
                cached = False
            packed_reused = False
            if value is None:
                if weight is None:
                    weight = source.tensor(
                        target.source_name,
                        row_start=target.row_start,
                        row_end=target.row_end,
                        device="cpu",
                    )
                encoded = None
                packed_path = None
                if packed_cache_dir is not None:
                    packed_path = dense_precision_cache_path(
                        root,
                        source.weight_map[target.source_name],
                        target,
                        precision,
                        packed_cache_dir,
                    )
                    encoded = load_dense_candidate(
                        packed_path,
                        name=target.name,
                        precision=precision,
                        shape=(target.rows, target.columns),
                    )
                if encoded is None and precision.nint_spec is not None:
                    legacy_packed = packed_index.get(
                        (
                            target.name,
                            precision.nint_spec.bits,
                            precision.nint_spec.groupsize,
                            precision.nint_spec.sub_bits,
                        )
                    )
                    if legacy_packed is not None:
                        encoded = _load_packed_candidate(
                            legacy_packed,
                            stats,
                            profile,
                            precision.nint_spec,
                        )
                if encoded is not None:
                    packed_reused = True
                if precision.nint_spec is not None:
                    if encoded is None:
                        encoded = _quantize(
                            weight,
                            precision.nint_spec,
                            backend=backend,
                            device=device,
                        )
                    value = evaluate_tensor_candidate(
                        weight,
                        stats,
                        profile,
                        precision.nint_spec,
                        backend=backend,
                        device=device,
                        row_chunk=row_chunk,
                        encoded=encoded,
                    )
                else:
                    value, encoded = evaluate_tensor_precision(
                        weight,
                        stats,
                        profile,
                        precision,
                        backend=backend,
                        device=device,
                        row_chunk=row_chunk,
                        encoded=encoded,
                        artifact_root=root,
                        quantization_importance=(
                            None
                            if quantization_importance is None
                            else quantization_importance[target.name]
                        ),
                    )
                if packed_path is not None and not packed_path.exists():
                    save_dense_candidate(
                        packed_path,
                        encoded,
                        name=target.name,
                        precision=precision,
                        shape=(target.rows, target.columns),
                    )
                _save_candidate_cache(path, value, identity)
            _materialize_scheme_candidate_cache(legacy_path, value, identity)
            result[name][profile] = value
            print(
                json.dumps(
                    {
                        "event": "candidate",
                        "done": done,
                        "total": total,
                        "tensor": name,
                        "profile": profile,
                        "cached": cached,
                        "packed_reused": packed_reused,
                        "train_nmse_percent": value.train_nmse_percent,
                        "validation_nmse_percent": value.validation_nmse_percent,
                        "elapsed_seconds": round(time.time() - started, 3),
                    }
                ),
                flush=True,
            )
        del weight
    return result


def allocate_scheme(
    evaluations: Mapping[str, Mapping[str, TensorCandidateEvaluation]],
    output: str | Path,
    *,
    target_profile: str,
    statistics: CalibrationStatistics,
    metadata: Mapping[str, Any] | None = None,
) -> CalibrationScheme:
    profile_name = target_profile.upper()
    groups: dict[str, list[str]] = {}
    for name, choices in evaluations.items():
        if profile_name not in choices:
            raise ValueError(f"tensor {name} has no {profile_name} candidate")
        groups.setdefault(choices[profile_name].group, []).append(name)

    group_candidates: list[GroupCandidate] = []
    for group, names in sorted(groups.items()):
        available = set.intersection(*(set(evaluations[name]) for name in names))
        for profile in sorted(available):
            values = [evaluations[name][profile] for name in names]
            group_candidates.append(
                GroupCandidate(
                    group=group,
                    profile=profile,
                    specs={value.name: value.descriptor for value in values},
                    storage_bits=sum(value.storage_bits for value in values),
                    train_loss=float(sum(value.train_loss for value in values)),
                    validation_loss=float(sum(value.validation_loss for value in values)),
                )
            )
    budget = sum(
        next(
            item.storage_bits
            for item in group_candidates
            if item.group == group and item.profile == profile_name
        )
        for group in groups
    )
    allocation = allocate(group_candidates, budget)

    selections: dict[str, TensorSelection] = {}
    for group, chosen in allocation.selected.items():
        for name, _descriptor in chosen.specs.items():
            value = evaluations[name][chosen.profile]
            selections[name] = TensorSelection(
                name=name,
                group=group,
                spec=value.spec,
                rows=value.rows,
                columns=value.columns,
                storage_bits=value.storage_bits,
                train_loss=value.train_loss,
                validation_loss=value.validation_loss,
                precision=value.descriptor,
            )
    candidate_table = {
        group: [
            {
                "profile": item.profile,
                "storage_bits": item.storage_bits,
                "train_loss": item.train_loss,
                "validation_loss": item.validation_loss,
                "precisions": {
                    name: precision_document(evaluations[name][item.profile].descriptor)
                    for name in sorted(item.specs)
                },
            }
            for item in group_candidates
            if item.group == group
        ]
        for group in sorted(groups)
    }
    scheme = CalibrationScheme(
        path=None,
        target_profile=profile_name,
        target_storage_bits=budget,
        selections=selections,
        metadata={
            "objective": "kfac_function_delta",
            "statistics": str(statistics.path),
            "allocator": allocation.solver,
            "allocation_train_loss": allocation.train_loss,
            "allocation_validation_loss": allocation.validation_loss,
            **dict(metadata or {}),
        },
        candidate_table=candidate_table,
    )
    save_scheme(output, scheme)
    from mfq.calibration.artifact import load_scheme

    return load_scheme(output)


def _layer_index(name: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", name)
    if match is None:
        raise ValueError(f"cannot determine layer index from tensor {name!r}")
    return int(match.group(1))


def _enumerate_layer_strategies(
    evaluations: Mapping[str, Mapping[str, TensorCandidateEvaluation]],
    scheme: CalibrationScheme,
) -> dict[int, list[LayerStrategy]]:
    """Enumerate every common-profile combination for each decoder layer."""

    names_by_layer: dict[int, list[str]] = {}
    for name in scheme.selections:
        if name not in evaluations:
            raise ValueError(f"calibration evaluations have no selected tensor {name}")
        names_by_layer.setdefault(_layer_index(name), []).append(name)

    result: dict[int, list[LayerStrategy]] = {}
    for layer, layer_names in sorted(names_by_layer.items()):
        names_by_group: dict[str, list[str]] = {}
        for name in layer_names:
            names_by_group.setdefault(scheme.selections[name].group, []).append(name)

        group_options: list[tuple[str, list[tuple[str, list[TensorCandidateEvaluation]]]]] = []
        for group, names in sorted(names_by_group.items()):
            profiles = set.intersection(*(set(evaluations[name]) for name in names))
            options = [
                (profile, [evaluations[name][profile] for name in sorted(names)])
                for profile in sorted(profiles)
            ]
            if not options:
                raise ValueError(f"precision group {group} has no common profiles")
            group_options.append((group, options))

        candidates: list[LayerStrategy] = []
        for combination in itertools.product(*(options for _group, options in group_options)):
            storage_bits = sum(
                value.storage_bits for _profile, values in combination for value in values
            )
            specs = {
                value.name: value.spec if value.spec is not None else value.descriptor
                for _profile, values in combination
                for value in values
            }
            profiles = {
                group: option[0]
                for (group, _options), option in zip(group_options, combination, strict=True)
            }
            candidates.append(
                LayerStrategy(
                    layer=layer,
                    name=",".join(
                        f"{group.rsplit('.', 1)[-1]}={profile}"
                        for group, profile in sorted(profiles.items())
                    ),
                    specs=specs,
                    profiles=profiles,
                    storage_bits=storage_bits,
                    train_loss=float(
                        sum(
                            value.train_loss for _profile, values in combination for value in values
                        )
                    ),
                    validation_loss=float(
                        sum(
                            value.validation_loss
                            for _profile, values in combination
                            for value in values
                        )
                    ),
                )
            )
        if not candidates:
            raise RuntimeError(f"layer {layer} has no common-profile strategy")
        result[layer] = candidates
    return result


def _base_layer_strategy(
    layer: int,
    candidates: Sequence[LayerStrategy],
    scheme: CalibrationScheme,
) -> LayerStrategy:
    names = candidates[0].specs
    base_specs = {
        name: (
            scheme.selections[name].spec
            if scheme.selections[name].spec is not None
            else scheme.selections[name].descriptor
        )
        for name in names
    }
    matches = [item for item in candidates if item.specs == base_specs]
    if len(matches) != 1:
        raise RuntimeError(f"base scheme has {len(matches)} matching layer {layer} strategies")
    return matches[0]


def _pareto_strategy_frontier(
    candidates: Sequence[LayerStrategy],
) -> list[LayerStrategy]:
    """Keep every strategy not dominated in storage and surrogate train loss."""

    ordered = sorted(
        candidates,
        key=lambda item: (item.storage_bits, item.train_loss, item.name),
    )
    frontier: list[LayerStrategy] = []
    best_loss = float("inf")
    for item in ordered:
        if item.train_loss < best_loss:
            frontier.append(item)
            best_loss = item.train_loss
    return frontier


def build_layer_strategies(
    evaluations: Mapping[str, Mapping[str, TensorCandidateEvaluation]],
    scheme: CalibrationScheme,
    *,
    top_k: int = 8,
) -> dict[int, list[LayerStrategy]]:
    """Retain low-surrogate strategies within each layer's current storage budget."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    enumerated = _enumerate_layer_strategies(evaluations, scheme)
    result: dict[int, list[LayerStrategy]] = {}
    for layer, candidates in enumerated.items():
        layer_names = candidates[0].specs
        budget = sum(scheme.selections[name].storage_bits for name in layer_names)
        feasible = [item for item in candidates if item.storage_bits <= budget]
        if not feasible:
            raise RuntimeError(f"layer {layer} has no strategy within its {budget}-bit budget")
        feasible.sort(key=lambda item: (item.train_loss, item.storage_bits, item.name))
        kept = feasible[:top_k]
        base = _base_layer_strategy(layer, candidates, scheme)
        if not any(item.specs == base.specs for item in kept):
            kept.append(base)
        result[layer] = kept
    return result


def build_global_layer_strategies(
    evaluations: Mapping[str, Mapping[str, TensorCandidateEvaluation]],
    scheme: CalibrationScheme,
) -> dict[int, list[LayerStrategy]]:
    """Build the exact one-precision-group neighborhood around the current scheme."""

    enumerated = _enumerate_layer_strategies(evaluations, scheme)
    result: dict[int, list[LayerStrategy]] = {}
    for layer, candidates in enumerated.items():
        base = _base_layer_strategy(layer, candidates, scheme)
        kept = [
            item
            for item in candidates
            if sum(item.profiles[group] != base.profiles[group] for group in base.profiles) <= 1
        ]
        result[layer] = sorted(
            kept,
            key=lambda item: (item.storage_bits, item.train_loss, item.name),
        )
    return result


def build_compensated_layer_strategies(
    evaluations: Mapping[str, Mapping[str, TensorCandidateEvaluation]],
    scheme: CalibrationScheme,
) -> dict[int, list[LayerStrategy]]:
    """Build complete per-layer surrogate Pareto sets for cumulative-budget greedy search."""

    enumerated = _enumerate_layer_strategies(evaluations, scheme)
    result: dict[int, list[LayerStrategy]] = {}
    for layer, candidates in enumerated.items():
        kept = _pareto_strategy_frontier(candidates)
        base = _base_layer_strategy(layer, candidates, scheme)
        if not any(item.specs == base.specs for item in kept):
            kept.append(base)
        result[layer] = sorted(
            kept,
            key=lambda item: (item.storage_bits, item.train_loss, item.name),
        )
    return result


__all__ = [
    "DENSE_PROFILE_FAMILIES",
    "NINT_CALIBRATION_PROFILES",
    "LayerStrategy",
    "TensorCandidateEvaluation",
    "allocate_scheme",
    "build_compensated_layer_strategies",
    "build_global_layer_strategies",
    "build_layer_strategies",
    "evaluate_candidates",
    "evaluate_tensor_candidate",
    "evaluate_tensor_precision",
    "load_scheme_candidate_evaluations",
    "nint_row_storage_bits",
    "nint_storage_bits",
]
