"""Packed heterogeneous precision candidates for layerwise calibration."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from mfq.calibration.artifact import (
    ExpertPrecision,
    nint_expert_precision,
    precision_document,
)
from mfq.formats.io import MfqTensor, pack_tensor_payload, unpack_tensor_payload
from mfq.formats.nint import NintSpec
from mfq.formats.nint8_zero import Nint8ZeroTensor
from mfq.formats.npq0_l import Npq0LTensor
from mfq.formats.npq0_s import Npq0STensor
from mfq.formats.nvq import NvqJscTensor, NvqTensor
from mfq.formats.nvq1_l import Nvq1LTensor
from mfq.formats.nvq1_s import Nvq1STensor
from mfq.quantize.expert_nint import (
    dequantize_flat_precision,
    quantize_flat_cohort,
    resolve_precision_artifact,
)
from mfq.quantize.nint_quant import NintTensor
from mfq.quantize.nint_quant import quantize as quantize_nint_cpu
from mfq.quantize.nint_quant_torch import quantize_axis0 as quantize_nint_accelerator
from mfq.runtime.torch_linear import (
    TorchNint8ZeroLinear,
    TorchNintLinear,
    TorchNvqLinear,
    is_nvq_tensor,
)

DensePrecision = NintSpec | ExpertPrecision
PackedDenseLinear = TorchNintLinear | TorchNint8ZeroLinear | TorchNvqLinear


def normalize_dense_precision(value: DensePrecision) -> ExpertPrecision:
    if isinstance(value, NintSpec):
        return nint_expert_precision(value)
    if isinstance(value, ExpertPrecision):
        return value
    raise TypeError(f"unsupported Dense precision descriptor: {type(value)!r}")


def dense_precision_cache_path(
    model_root: str | Path,
    source_shard: str | Path,
    target,
    precision: DensePrecision,
    cache_dir: str | Path,
) -> Path:
    """Return the stable packed-cache path for one Dense candidate."""

    root = Path(model_root).resolve()
    shard = Path(source_shard)
    shard = shard if shard.is_absolute() else root / shard
    descriptor = normalize_dense_precision(precision)
    stat = shard.stat()
    identity: dict[str, object] = {
        "root": str(root),
        "source": str(target.source_name),
        "row_start": int(target.row_start),
        "row_end": int(target.row_end),
        "precision": precision_document(descriptor),
        "source_bytes": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
    }
    if descriptor.artifact is not None:
        artifact = Path(descriptor.artifact)
        if not artifact.is_absolute():
            artifact = root / artifact
        artifact = artifact.resolve()
        artifact_stat = artifact.stat()
        identity["artifact"] = {
            "path": str(artifact),
            "bytes": int(artifact_stat.st_size),
            "mtime_ns": int(artifact_stat.st_mtime_ns),
        }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return Path(cache_dir).resolve() / f"{digest}.dense-packed.npz"


def quantize_dense_precision(
    weight: torch.Tensor,
    precision: DensePrecision,
    *,
    importance: np.ndarray | torch.Tensor | None = None,
    backend: str = "cuda",
    device: str | torch.device = "cuda:0",
    artifact_root: str | Path | None = None,
) -> MfqTensor:
    """Quantize one matrix into a compact Dense candidate."""

    descriptor = normalize_dense_precision(precision)
    if descriptor.nint_spec is not None:
        spec = descriptor.nint_spec
        if backend == "cpu":
            return quantize_nint_cpu(
                weight.detach().float().cpu().numpy(),
                spec,
                axis=0,
                importance=(
                    None
                    if importance is None or spec.bits not in {2, 3, 4, 5, 6}
                    else np.asarray(
                        importance.detach().cpu().numpy()
                        if isinstance(importance, torch.Tensor)
                        else importance,
                        dtype=np.float32,
                    )
                ),
            )
        if backend not in {"cuda", "metal"}:
            raise ValueError("Dense candidate backend must be cuda, metal, or cpu")
        return quantize_nint_accelerator(
            weight,
            spec,
            device=device,
            importance=(importance if spec.bits in {2, 3, 4, 5, 6} else None),
        )

    artifact = resolve_precision_artifact(
        descriptor,
        artifact_root=artifact_root,
    )
    quant_device: str | torch.device = "cpu" if backend == "cpu" else device
    result = quantize_flat_cohort(
        weight.detach().float().cpu().numpy(),
        descriptor,
        artifact=artifact,
        importance=importance,
        device=quant_device,
    )
    if not isinstance(result, Nint8ZeroTensor) and not is_nvq_tensor(result):
        raise TypeError(
            f"Dense calibration supports packed NINT/NVQ/NPQ candidates, got "
            f"{type(result).__name__} for {descriptor.family}"
        )
    return result


def packed_dense_linear(
    tensor: MfqTensor,
    device: str | torch.device,
) -> PackedDenseLinear:
    if isinstance(tensor, NintTensor):
        return TorchNintLinear(tensor, device)
    if torch.device(device).type != "cuda":
        raise RuntimeError("packed NVQ/NPQ/NINT8-0 Dense calibration currently requires CUDA")
    if isinstance(tensor, Nint8ZeroTensor):
        return TorchNint8ZeroLinear(tensor, device)
    if is_nvq_tensor(tensor):
        return TorchNvqLinear(tensor, device)  # type: ignore[arg-type]
    raise TypeError(f"unsupported packed Dense candidate: {type(tensor).__name__}")


def dense_candidate_storage_bits(tensor: MfqTensor) -> int:
    _dtype, payload = pack_tensor_payload(tensor)
    return len(payload) * 8


def dense_candidate_reconstruction(tensor: MfqTensor) -> np.ndarray:
    return dequantize_flat_precision(tensor)


def slice_dense_tensor_rows(
    tensor: MfqTensor,
    start: int,
    end: int,
) -> MfqTensor:
    """Return an exact packed row slice without reconstructing or requantizing."""

    shape = tuple(int(value) for value in getattr(tensor, "shape", ()))
    if len(shape) != 2 or int(getattr(tensor, "axis", -1)) != 0:
        raise ValueError("Dense packed row slicing requires a 2D axis-0 tensor")
    start, end = int(start), int(end)
    if start < 0 or end <= start or end > shape[0]:
        raise IndexError(f"invalid Dense packed row slice {start}:{end} of {shape[0]}")
    result_shape = (end - start, shape[1])

    def rows(value: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(value)[start:end]).copy()

    if isinstance(tensor, NintTensor):
        return replace(
            tensor,
            shape=result_shape,
            q=rows(tensor.q),
            neuron_scale=rows(tensor.neuron_scale),
            neuron_min=rows(tensor.neuron_min),
            sub_scale=rows(tensor.sub_scale),
            sub_min=rows(tensor.sub_min),
        )
    if isinstance(tensor, Nint8ZeroTensor):
        return replace(
            tensor,
            shape=result_shape,
            scale=rows(tensor.scale),
            q=rows(tensor.q),
        )
    if isinstance(tensor, NvqTensor):
        return replace(
            tensor,
            shape=result_shape,
            neuron_scale=rows(tensor.neuron_scale),
            sub_scale=rows(tensor.sub_scale),
            indices=rows(tensor.indices),
            signs=rows(tensor.signs),
        )
    if isinstance(tensor, NvqJscTensor):
        return replace(
            tensor,
            shape=result_shape,
            neuron_scale=rows(tensor.neuron_scale),
            state=rows(tensor.state),
            indices=rows(tensor.indices),
            signs=rows(tensor.signs),
        )
    if isinstance(tensor, (Npq0LTensor, Npq0STensor)):
        return replace(
            tensor,
            shape=result_shape,
            neuron_scale=rows(tensor.neuron_scale),
            state=rows(tensor.state),
            indices=rows(tensor.indices),
        )
    if isinstance(tensor, (Nvq1LTensor, Nvq1STensor)):
        return replace(
            tensor,
            shape=result_shape,
            neuron_scale=rows(tensor.neuron_scale),
            sub_scale=rows(tensor.sub_scale),
            indices=rows(tensor.indices),
            delta_sign=rows(tensor.delta_sign),
        )
    raise TypeError(f"unsupported Dense packed row slice: {type(tensor).__name__}")


def save_dense_candidate(
    path: str | Path,
    tensor: MfqTensor,
    *,
    name: str,
    precision: DensePrecision,
    shape: tuple[int, int],
) -> None:
    output = Path(path)
    if output.exists():
        return
    descriptor = normalize_dense_precision(precision)
    dtype, payload = pack_tensor_payload(tensor)
    document = {
        "format": "mfq.dense-packed-candidate.v1",
        "name": name,
        "precision": precision_document(descriptor),
        "shape": [int(shape[0]), int(shape[1])],
        "dtype": dtype,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    metadata = np.frombuffer(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    )
    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Dense candidate temporary already exists: {temporary}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as stream:
        np.savez(
            stream,
            metadata=metadata,
            payload=np.frombuffer(payload, dtype=np.uint8),
        )
    os.replace(temporary, output)


def load_dense_candidate(
    path: str | Path,
    *,
    name: str,
    precision: DensePrecision,
    shape: tuple[int, int],
) -> MfqTensor | None:
    source = Path(path)
    if not source.is_file():
        return None
    descriptor = normalize_dense_precision(precision)
    with np.load(source, allow_pickle=False) as archive:
        document = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        payload = archive["payload"].tobytes()
    expected = {
        "format": "mfq.dense-packed-candidate.v1",
        "name": name,
        "precision": precision_document(descriptor),
        "shape": [int(shape[0]), int(shape[1])],
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError(f"invalid Dense packed candidate metadata: {source}")
    if hashlib.sha256(payload).hexdigest() != document.get("payload_sha256"):
        raise ValueError(f"Dense packed candidate payload digest mismatch: {source}")
    tensor = unpack_tensor_payload(str(document["dtype"]), payload)
    tensor_shape = tuple(int(value) for value in getattr(tensor, "shape", ()))
    if tensor_shape != shape:
        raise ValueError(f"Dense packed candidate shape mismatch: {source}")
    return tensor


__all__ = [
    "DensePrecision",
    "PackedDenseLinear",
    "dense_candidate_reconstruction",
    "dense_candidate_storage_bits",
    "dense_precision_cache_path",
    "load_dense_candidate",
    "normalize_dense_precision",
    "packed_dense_linear",
    "quantize_dense_precision",
    "save_dense_candidate",
    "slice_dense_tensor_rows",
]
