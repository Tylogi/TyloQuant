"""Disk-backed dual-path layerwise calibration replay."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from mfq.calibration.dataset import CalibrationBatch, CalibrationCorpus


@dataclass(frozen=True)
class HiddenTrace:
    layer: int
    reference_energy: float
    quantized_energy: float
    squared_error: float
    dot_product: float
    value_count: int

    @property
    def relative_rmse_percent(self) -> float:
        return 100.0 * (self.squared_error / max(self.reference_energy, 1e-300)) ** 0.5

    @property
    def cosine_similarity(self) -> float:
        denominator = max(
            (self.reference_energy * self.quantized_energy) ** 0.5,
            1e-300,
        )
        return self.dot_product / denominator

    def document(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "relative_rmse_percent": self.relative_rmse_percent,
            "cosine_similarity": self.cosine_similarity,
        }


class LayerwiseBackend(Protocol):
    num_layers: int
    hidden_size: int
    teacher_dtype: torch.dtype
    quantized_dtype: torch.dtype

    def initial_hidden(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    def release_initial_state(self) -> None: ...

    def layer(self, layer_index: int, *, quantized: bool) -> AbstractContextManager[Any]: ...

    def forward_layer(
        self,
        layer: Any,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor: ...


class HiddenStateStore:
    """A fixed-size mmap that preserves FP16 or BF16 tensor bits."""

    def __init__(
        self,
        path: str | Path,
        token_count: int,
        hidden_size: int,
        dtype: torch.dtype,
        *,
        create: bool = True,
    ) -> None:
        self.path = Path(path).resolve()
        self.token_count = int(token_count)
        self.hidden_size = int(hidden_size)
        self.dtype = dtype
        if self.token_count <= 0 or self.hidden_size <= 0:
            raise ValueError("hidden-state mmap dimensions must be positive")
        if dtype == torch.bfloat16:
            numpy_dtype = np.uint16
        elif dtype == torch.float16:
            numpy_dtype = np.float16
        elif dtype == torch.float32:
            numpy_dtype = np.float32
        else:
            raise ValueError(f"unsupported hidden-state dtype: {dtype}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w+" if create else "r+"
        self.array = np.memmap(
            self.path,
            mode=mode,
            dtype=numpy_dtype,
            shape=(self.token_count, self.hidden_size),
        )

    def write(self, start: int, value: torch.Tensor) -> None:
        matrix = value.detach().reshape(-1, value.shape[-1])
        if matrix.shape[1] != self.hidden_size:
            raise ValueError(
                f"hidden width {matrix.shape[1]} does not match mmap width {self.hidden_size}"
            )
        end = start + int(matrix.shape[0])
        if start < 0 or end > self.token_count:
            raise IndexError(f"hidden write {start}:{end} exceeds {self.token_count} tokens")
        if self.dtype == torch.bfloat16:
            data = (
                matrix.to(device="cpu", dtype=torch.bfloat16)
                .contiguous()
                .view(torch.uint16)
                .numpy()
            )
        else:
            data = matrix.to(device="cpu", dtype=self.dtype).contiguous().numpy()
        self.array[start:end] = data

    def read(
        self,
        start: int,
        end: int,
        shape: Sequence[int],
        *,
        device: str | torch.device,
    ) -> torch.Tensor:
        if start < 0 or end < start or end > self.token_count:
            raise IndexError(f"hidden read {start}:{end} exceeds {self.token_count} tokens")
        data = np.array(self.array[start:end], copy=True)
        value = torch.from_numpy(data)
        if self.dtype == torch.bfloat16:
            value = value.view(torch.bfloat16)
        value = value.reshape(*shape, self.hidden_size)
        return value.to(device=device)

    def flush(self) -> None:
        self.array.flush()

    def close(self) -> None:
        self.flush()
        del self.array


def _batch_layout(batches: Sequence[CalibrationBatch]) -> list[tuple[CalibrationBatch, int, int]]:
    result: list[tuple[CalibrationBatch, int, int]] = []
    cursor = 0
    for batch in batches:
        count = int(batch.input_ids.size)
        result.append((batch, cursor, cursor + count))
        cursor += count
    return result


def _trace_batch(reference: torch.Tensor, quantized: torch.Tensor) -> tuple[float, ...]:
    reference = reference.float()
    quantized = quantized.float()
    difference = quantized - reference
    return (
        float(reference.square().sum(dtype=torch.float64).item()),
        float(quantized.square().sum(dtype=torch.float64).item()),
        float(difference.square().sum(dtype=torch.float64).item()),
        float((reference * quantized).sum(dtype=torch.float64).item()),
        float(reference.numel()),
    )


def validate_layerwise(
    backend: LayerwiseBackend,
    corpus: CalibrationCorpus,
    output: str | Path,
    *,
    work_dir: str | Path,
    split: str = "validation",
    window_length: int = 2048,
    batch_size: int = 1,
    max_tokens: int = 16_384,
    seed: int | None = None,
    keep_hidden: bool = False,
) -> list[HiddenTrace]:
    """Replay a teacher and cumulative quantized path one layer at a time."""

    report_path = Path(output).resolve()
    if report_path.exists():
        raise FileExistsError(f"layerwise report already exists: {report_path}")
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    teacher_path = work / "teacher-hidden.bin"
    quantized_path = work / "quantized-hidden.bin"
    for path in (teacher_path, quantized_path):
        if path.exists():
            raise FileExistsError(f"layerwise hidden-state file already exists: {path}")

    batches = list(
        corpus.iter_batches(
            split,
            window_length=window_length,
            batch_size=batch_size,
            max_tokens=max_tokens,
            seed=seed,
        )
    )
    if not batches:
        raise ValueError(f"calibration corpus produced no {split} batches")
    layout = _batch_layout(batches)
    token_count = layout[-1][2]
    teacher_store = HiddenStateStore(
        teacher_path,
        token_count,
        backend.hidden_size,
        backend.teacher_dtype,
    )
    quantized_store = HiddenStateStore(
        quantized_path,
        token_count,
        backend.hidden_size,
        backend.quantized_dtype,
    )
    device = getattr(backend, "device", "cuda")
    traces: list[HiddenTrace] = []
    try:
        for batch, start, _end in layout:
            ids = torch.as_tensor(batch.input_ids, device=device, dtype=torch.int64)
            hidden = backend.initial_hidden(ids)
            teacher_store.write(start, hidden)
            quantized_store.write(start, hidden)
        backend.release_initial_state()
        teacher_store.flush()
        quantized_store.flush()

        for layer_index in range(backend.num_layers):
            with backend.layer(layer_index, quantized=False) as layer:
                for batch, start, end in layout:
                    shape = batch.input_ids.shape
                    hidden = teacher_store.read(start, end, shape, device=device)
                    output_value = backend.forward_layer(layer, layer_index, hidden)
                    teacher_store.write(start, output_value)
            teacher_store.flush()

            totals = np.zeros(5, dtype=np.float64)
            with backend.layer(layer_index, quantized=True) as layer:
                for batch, start, end in layout:
                    shape = batch.input_ids.shape
                    hidden = quantized_store.read(start, end, shape, device=device)
                    output_value = backend.forward_layer(layer, layer_index, hidden)
                    reference = teacher_store.read(start, end, shape, device=device)
                    totals += np.asarray(_trace_batch(reference, output_value), dtype=np.float64)
                    quantized_store.write(start, output_value)
            quantized_store.flush()
            trace = HiddenTrace(
                layer=layer_index,
                reference_energy=float(totals[0]),
                quantized_energy=float(totals[1]),
                squared_error=float(totals[2]),
                dot_product=float(totals[3]),
                value_count=int(totals[4]),
            )
            traces.append(trace)
            print(json.dumps({"event": "layerwise", **trace.document()}), flush=True)

        document = {
            "format": "mfq.layerwise-validation.v1",
            "corpus": str(corpus.root),
            "split": split,
            "window_length": int(window_length),
            "batch_size": int(batch_size),
            "token_count": int(token_count),
            "teacher_dtype": str(backend.teacher_dtype),
            "quantized_dtype": str(backend.quantized_dtype),
            "layers": [trace.document() for trace in traces],
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, report_path)
    finally:
        backend.release_initial_state()
        teacher_store.close()
        quantized_store.close()
        if not keep_hidden:
            teacher_path.unlink(missing_ok=True)
            quantized_path.unlink(missing_ok=True)
    return traces


__all__ = [
    "HiddenStateStore",
    "HiddenTrace",
    "LayerwiseBackend",
    "validate_layerwise",
]
