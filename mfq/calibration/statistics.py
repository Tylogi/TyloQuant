"""Activation and loss-sensitivity statistics for calibration.

Input statistics are diagonal second moments ``E[x_i^2]``.  Output-neuron
statistics use a Rademacher estimate of the diagonal Fisher of next-token NLL.
Their product is a Kronecker-factored approximation to function-level weight
perturbation loss and naturally includes the SwiGLU product derivatives.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from mfq.calibration.collector import HiddenStateStore
from mfq.calibration.dataset import CalibrationBatch, CalibrationCorpus
from mfq.calibration.qwen35 import Qwen35LinearTarget, qwen35_linear_targets

_FORMAT = "mfq.calibration-statistics.v1"
_METADATA_KEY = "__metadata_json__"


@dataclass(frozen=True)
class TensorStatistics:
    target: Qwen35LinearTarget
    train_input_second_moment: np.ndarray
    validation_input_second_moment: np.ndarray
    train_row_fisher: np.ndarray
    validation_row_fisher: np.ndarray
    train_input_count: int
    validation_input_count: int
    train_fisher_probes: int
    validation_fisher_probes: int


@dataclass(frozen=True)
class CalibrationStatistics:
    path: Path
    entries: dict[str, TensorStatistics]
    metadata: dict[str, Any]

    def require(self, name: str) -> TensorStatistics:
        try:
            return self.entries[name]
        except KeyError as exc:
            raise KeyError(f"calibration statistics have no tensor {name!r}") from exc


def _positive_floor(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not array.size or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"invalid non-negative statistic for {label}")
    positive = array[array > 0]
    if not positive.size:
        raise ValueError(f"calibration statistic is entirely zero for {label}")
    floor = max(float(positive.mean()) * 1e-12, np.finfo(np.float32).tiny)
    return np.ascontiguousarray(np.maximum(array, floor), dtype=np.float32)


def save_statistics(
    path: str | Path,
    entries: Mapping[str, TensorStatistics],
    metadata: Mapping[str, Any],
) -> None:
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"calibration statistics already exist: {output}")
    if not entries:
        raise ValueError("cannot save empty calibration statistics")
    output.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    entry_metadata: dict[str, Any] = {}
    for index, (name, entry) in enumerate(sorted(entries.items())):
        if name != entry.target.name:
            raise ValueError(f"statistics key {name!r} disagrees with target {entry.target.name!r}")
        prefix = f"entry_{index:04d}"
        values = {
            "train_input": entry.train_input_second_moment,
            "validation_input": entry.validation_input_second_moment,
            "train_fisher": entry.train_row_fisher,
            "validation_fisher": entry.validation_row_fisher,
        }
        for suffix, raw in values.items():
            arrays[f"{prefix}_{suffix}"] = _positive_floor(raw, f"{name}/{suffix}")
        entry_metadata[name] = {
            "prefix": prefix,
            "target": asdict(entry.target),
            "counts": {
                "train_input": int(entry.train_input_count),
                "validation_input": int(entry.validation_input_count),
                "train_fisher_probes": int(entry.train_fisher_probes),
                "validation_fisher_probes": int(entry.validation_fisher_probes),
            },
        }

    document = {"format": _FORMAT, "metadata": dict(metadata), "entries": entry_metadata}
    arrays[_METADATA_KEY] = np.frombuffer(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    ).copy()
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, output)


def load_statistics(path: str | Path) -> CalibrationStatistics:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"calibration statistics do not exist: {resolved}")
    with np.load(resolved, allow_pickle=False) as archive:
        if _METADATA_KEY not in archive.files:
            raise ValueError(f"calibration statistics have no metadata: {resolved}")
        try:
            document = json.loads(archive[_METADATA_KEY].tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid calibration statistics metadata: {resolved}") from exc
        if document.get("format") != _FORMAT:
            raise ValueError(
                f"unsupported calibration statistics format: {document.get('format')!r}"
            )
        entries: dict[str, TensorStatistics] = {}
        for name, item in document.get("entries", {}).items():
            prefix = str(item["prefix"])
            target = Qwen35LinearTarget(**item["target"])
            arrays = {
                suffix: _positive_floor(archive[f"{prefix}_{suffix}"], f"{name}/{suffix}")
                for suffix in (
                    "train_input",
                    "validation_input",
                    "train_fisher",
                    "validation_fisher",
                )
            }
            if arrays["train_input"].shape != (target.columns,):
                raise ValueError(f"input-stat width mismatch for {name}")
            if arrays["validation_input"].shape != (target.columns,):
                raise ValueError(f"validation input-stat width mismatch for {name}")
            if arrays["train_fisher"].shape != (target.rows,):
                raise ValueError(f"row-Fisher width mismatch for {name}")
            if arrays["validation_fisher"].shape != (target.rows,):
                raise ValueError(f"validation row-Fisher width mismatch for {name}")
            counts = item["counts"]
            entries[str(name)] = TensorStatistics(
                target=target,
                train_input_second_moment=arrays["train_input"],
                validation_input_second_moment=arrays["validation_input"],
                train_row_fisher=arrays["train_fisher"],
                validation_row_fisher=arrays["validation_fisher"],
                train_input_count=int(counts["train_input"]),
                validation_input_count=int(counts["validation_input"]),
                train_fisher_probes=int(counts["train_fisher_probes"]),
                validation_fisher_probes=int(counts["validation_fisher_probes"]),
            )
    if not entries:
        raise ValueError(f"calibration statistics contain no entries: {resolved}")
    return CalibrationStatistics(resolved, entries, dict(document.get("metadata", {})))


class Qwen35StatisticsCollector:
    """Hooks Qwen3.5 linears while preserving the model's original graph."""

    def __init__(self, targets: Sequence[Qwen35LinearTarget], device: torch.device) -> None:
        self.targets = tuple(targets)
        self.device = device
        self.by_module: dict[str, list[Qwen35LinearTarget]] = defaultdict(list)
        for target in targets:
            self.by_module[target.module_name].append(target)
        self.input_sums = {
            split: {
                module: torch.zeros(items[0].columns, device=device, dtype=torch.float64)
                for module, items in self.by_module.items()
            }
            for split in ("train", "validation")
        }
        self.input_counts = {
            split: {module: 0 for module in self.by_module} for split in ("train", "validation")
        }
        self.fisher_sums = {
            split: {
                target.name: torch.zeros(target.rows, device=device, dtype=torch.float64)
                for target in targets
            }
            for split in ("train", "validation")
        }
        self.fisher_probes = {"train": 0, "validation": 0}
        self.current_split = "train"
        self.mode = "idle"
        self.handles: list[Any] = []

    def _install_module(
        self,
        module: torch.nn.Module,
        module_name: str,
        targets: Sequence[Qwen35LinearTarget],
    ) -> None:
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"calibration target is not Linear: {module_name}")
        if module.in_features != targets[0].columns:
            raise ValueError(f"input width mismatch for {module_name}")
        if max(target.row_end for target in targets) > module.out_features:
            raise ValueError(f"output rows exceed module width for {module_name}")

        def pre_hook(_module, inputs, *, _name=module_name):
            if self.mode != "input":
                return
            value = inputs[0]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"non-tensor input from {_name}")
            matrix = value.detach().reshape(-1, value.shape[-1]).float()
            batch_sum = matrix.square().sum(dim=0)
            self.input_sums[self.current_split][_name].add_(batch_sum.to(torch.float64))
            self.input_counts[self.current_split][_name] += int(matrix.shape[0])

        def forward_hook(_module, _inputs, output, *, _targets=tuple(targets)):
            if self.mode != "fisher":
                return
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"non-tensor output from {_targets[0].module_name}")

            def add_gradient(gradient: torch.Tensor) -> None:
                matrix = gradient.detach().reshape(-1, gradient.shape[-1]).float()
                for target in _targets:
                    value = matrix[:, target.row_start : target.row_end]
                    row_sum = value.square().sum(dim=0)
                    self.fisher_sums[self.current_split][target.name].add_(
                        row_sum.to(torch.float64)
                    )

            output.register_hook(add_gradient)

        self.handles.append(module.register_forward_pre_hook(pre_hook))
        self.handles.append(module.register_forward_hook(forward_hook))

    def install(self, model: torch.nn.Module) -> None:
        modules = dict(model.named_modules())
        missing = sorted(set(self.by_module) - set(modules))
        if missing:
            raise ValueError(f"Qwen3.5 model is missing calibration modules: {missing[:8]}")
        for module_name, targets in self.by_module.items():
            self._install_module(modules[module_name], module_name, targets)

    def install_layer(self, layer: torch.nn.Module, layer_index: int) -> None:
        prefix = f"model.layers.{layer_index}."
        modules = dict(layer.named_modules())
        installed = 0
        for module_name, targets in self.by_module.items():
            if not module_name.startswith(prefix):
                continue
            local_name = module_name[len(prefix) :]
            try:
                module = modules[local_name]
            except KeyError as exc:
                raise ValueError(
                    f"Qwen3.5 layer {layer_index} is missing calibration module {local_name}"
                ) from exc
            self._install_module(module, module_name, targets)
            installed += 1
        if installed == 0:
            raise ValueError(f"Qwen3.5 layer {layer_index} has no calibration targets")

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def entries(self) -> dict[str, TensorStatistics]:
        result: dict[str, TensorStatistics] = {}
        for target in self.targets:
            input_values: dict[str, np.ndarray] = {}
            fisher_values: dict[str, np.ndarray] = {}
            for split in ("train", "validation"):
                input_count = self.input_counts[split][target.module_name]
                probes = self.fisher_probes[split]
                if input_count <= 0 or probes <= 0:
                    raise RuntimeError(
                        f"incomplete calibration statistics for {target.name}/{split}: "
                        f"input_count={input_count}, fisher_probes={probes}"
                    )
                input_values[split] = (
                    (self.input_sums[split][target.module_name] / float(input_count)).cpu().numpy()
                )
                fisher_values[split] = (
                    (self.fisher_sums[split][target.name] / float(probes)).cpu().numpy()
                )
            result[target.name] = TensorStatistics(
                target=target,
                train_input_second_moment=_positive_floor(input_values["train"], target.name),
                validation_input_second_moment=_positive_floor(
                    input_values["validation"], target.name
                ),
                train_row_fisher=_positive_floor(fisher_values["train"], target.name),
                validation_row_fisher=_positive_floor(fisher_values["validation"], target.name),
                train_input_count=self.input_counts["train"][target.module_name],
                validation_input_count=self.input_counts["validation"][target.module_name],
                train_fisher_probes=self.fisher_probes["train"],
                validation_fisher_probes=self.fisher_probes["validation"],
            )
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_identity(root: Path) -> dict[str, Any]:
    files = []
    for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
        path = root / name
        if path.is_file():
            files.append({"name": name, "size": path.stat().st_size, "sha256": _sha256(path)})
    shards = sorted(root.glob("*.safetensors"))
    return {
        "path": str(root),
        "metadata_files": files,
        "shards": [{"name": path.name, "size": path.stat().st_size} for path in shards],
    }


@dataclass(frozen=True)
class _BatchSlice:
    split: str
    batch: CalibrationBatch
    start: int
    end: int


def _statistics_layout(
    corpus: CalibrationCorpus,
    *,
    window_length: int,
    token_limits: Mapping[str, int],
    seed: int,
    seed_offsets: Mapping[str, int],
    drop_last: bool,
) -> tuple[list[_BatchSlice], dict[str, int]]:
    layout: list[_BatchSlice] = []
    token_counts: dict[str, int] = {}
    cursor = 0
    for split in ("train", "validation"):
        split_start = cursor
        for batch in corpus.iter_batches(
            split,
            window_length=window_length,
            batch_size=1,
            max_tokens=token_limits[split],
            seed=seed + seed_offsets[split],
            drop_last=drop_last,
        ):
            count = int(batch.input_ids.size)
            layout.append(_BatchSlice(split, batch, cursor, cursor + count))
            cursor += count
        token_counts[split] = cursor - split_start
        if token_counts[split] == 0:
            raise RuntimeError(f"no {split} calibration tokens were selected")
    return layout, token_counts


def _require_new_paths(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"calibration work files already exist: {existing[:4]}")


def _seed_hidden_store(
    backend: Any, layout: Sequence[_BatchSlice], store: HiddenStateStore
) -> None:
    try:
        for item in layout:
            input_ids = torch.as_tensor(item.batch.input_ids, dtype=torch.int64)
            store.write(item.start, backend.initial_hidden(input_ids))
        store.flush()
    finally:
        backend.release_initial_state()


def _collect_input_statistics(
    backend: Any,
    collector: Qwen35StatisticsCollector,
    corpus: CalibrationCorpus,
    work: Path,
    *,
    window_length: int,
    token_limits: Mapping[str, int],
    seed: int,
    keep_hidden: bool,
    emit: Callable[..., None],
) -> dict[str, int]:
    layout, token_counts = _statistics_layout(
        corpus,
        window_length=window_length,
        token_limits=token_limits,
        seed=seed,
        seed_offsets={"train": 0, "validation": 1},
        drop_last=False,
    )
    hidden_path = work / "input-hidden.bin"
    _require_new_paths([hidden_path])
    store = HiddenStateStore(
        hidden_path,
        layout[-1].end,
        backend.hidden_size,
        backend.teacher_dtype,
    )
    try:
        _seed_hidden_store(backend, layout, store)
        collector.mode = "input"
        for layer_index in range(backend.num_layers):
            with backend.layer(layer_index, quantized=False) as layer:
                collector.install_layer(layer, layer_index)
                try:
                    for item in layout:
                        collector.current_split = item.split
                        hidden = store.read(
                            item.start,
                            item.end,
                            item.batch.input_ids.shape,
                            device=backend.device,
                        )
                        store.write(
                            item.start,
                            backend.forward_layer(layer, layer_index, hidden),
                        )
                finally:
                    collector.close()
            store.flush()
            emit("input_forward", layer_index + 1, backend.num_layers, tokens=token_counts)
    finally:
        collector.mode = "idle"
        collector.close()
        backend.release_initial_state()
        store.close()
        if not keep_hidden:
            hidden_path.unlink(missing_ok=True)
    return token_counts


def _load_final_norm(backend: Any) -> torch.nn.Module:
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Qwen3.5 calibration requires transformers") from exc
    with torch.device("meta"):
        norm = Qwen3_5RMSNorm(backend.hidden_size, eps=float(backend.config.rms_norm_eps))
    result = norm.load_state_dict(
        {
            "weight": backend.index.tensor(
                "model.language_model.norm.weight",
                device="cpu",
            )
        },
        strict=True,
        assign=True,
    )
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Qwen3.5 final norm state mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    return norm.to(device=backend.device, dtype=backend.teacher_dtype).eval().requires_grad_(False)


def _chunked_head_gradient(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    final_norm: torch.nn.Module,
    head_weight: torch.Tensor,
    generator: torch.Generator,
    *,
    row_chunk: int,
) -> tuple[torch.Tensor, float]:
    value = hidden_states.detach().requires_grad_(True)
    normalized = final_norm(value[:, :-1, :])
    matrix = normalized.reshape(-1, normalized.shape[-1])
    labels = input_ids[:, 1:].reshape(-1)
    rows, columns = (int(head_weight.shape[0]), int(head_weight.shape[1]))
    if row_chunk == 0:
        row_chunk = rows
    if row_chunk < 0:
        raise ValueError("lm_head row chunk must be non-negative")
    if columns != matrix.shape[1]:
        raise ValueError(f"lm_head width {columns} does not match hidden width {matrix.shape[1]}")
    if row_chunk >= rows:
        weight = head_weight.to(device=matrix.device, dtype=matrix.dtype)
        logits = functional.linear(matrix, weight)
        losses = functional.cross_entropy(
            logits.float(),
            labels,
            reduction="none",
        )
        signs = torch.empty_like(losses)
        signs.bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
        probe = (losses * signs).sum() / math.sqrt(float(losses.numel()))
        probe_value = float(probe.detach().item())
        probe.backward()
        gradient = value.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise FloatingPointError("non-finite lm_head hidden gradient")
        return gradient.detach(), probe_value

    maximum = torch.full(
        (matrix.shape[0],),
        -torch.inf,
        device=matrix.device,
        dtype=torch.float32,
    )
    denominator = torch.zeros_like(maximum)
    target_logits = torch.full_like(maximum, torch.nan)
    with torch.no_grad():
        for start in range(0, rows, row_chunk):
            end = min(start + row_chunk, rows)
            weight = head_weight[start:end].to(
                device=matrix.device,
                dtype=matrix.dtype,
            )
            logits = functional.linear(matrix.detach(), weight).float()
            local_maximum = logits.max(dim=1).values
            new_maximum = torch.maximum(maximum, local_maximum)
            denominator = denominator * torch.exp(maximum - new_maximum)
            denominator += torch.exp(logits - new_maximum[:, None]).sum(dim=1)
            maximum = new_maximum
            selected = (labels >= start) & (labels < end)
            if selected.any():
                target_logits[selected] = logits[
                    selected,
                    labels[selected] - start,
                ]
            del weight, logits
    if (
        not torch.isfinite(maximum).all()
        or not torch.isfinite(denominator).all()
        or not torch.isfinite(target_logits).all()
        or torch.any(denominator <= 0)
    ):
        raise FloatingPointError("non-finite chunked lm_head softmax state")

    losses = maximum + denominator.log() - target_logits
    signs = torch.empty_like(losses)
    signs.bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
    scales = signs / math.sqrt(float(losses.numel()))
    probe_value = float((losses * scales).sum().item())
    for start in range(0, rows, row_chunk):
        end = min(start + row_chunk, rows)
        weight = head_weight[start:end].to(
            device=matrix.device,
            dtype=matrix.dtype,
        )
        logits = functional.linear(matrix, weight)
        coefficients = torch.exp(logits.float() - maximum[:, None]) / denominator[:, None]
        coefficients *= scales[:, None]
        selected = (labels >= start) & (labels < end)
        if selected.any():
            coefficients[selected, labels[selected] - start] -= scales[selected]
        torch.autograd.backward(
            logits.float(),
            coefficients,
            retain_graph=end < rows,
        )
        del weight, logits, coefficients
    gradient = value.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise FloatingPointError("non-finite chunked lm_head hidden gradient")
    return gradient.detach(), probe_value


def _collect_fisher_statistics(
    backend: Any,
    collector: Qwen35StatisticsCollector,
    corpus: CalibrationCorpus,
    work: Path,
    *,
    window_length: int,
    token_limits: Mapping[str, int],
    head_row_chunk: int,
    seed: int,
    keep_hidden: bool,
    emit: Callable[..., None],
) -> tuple[dict[str, int], dict[str, float]]:
    layout, token_counts = _statistics_layout(
        corpus,
        window_length=window_length,
        token_limits=token_limits,
        seed=seed,
        seed_offsets={"train": 2, "validation": 3},
        drop_last=True,
    )
    hidden_paths = [
        work / f"fisher-hidden-{index:03d}.bin" for index in range(backend.num_layers + 1)
    ]
    gradient_paths = [work / "fisher-gradient-a.bin", work / "fisher-gradient-b.bin"]
    all_paths = [*hidden_paths, *gradient_paths]
    _require_new_paths(all_paths)
    hidden_stores: list[HiddenStateStore] = []
    gradient_stores: list[HiddenStateStore] = []
    probe_sums = {"train": 0.0, "validation": 0.0}
    try:
        for path in hidden_paths:
            hidden_stores.append(
                HiddenStateStore(
                    path,
                    layout[-1].end,
                    backend.hidden_size,
                    backend.teacher_dtype,
                )
            )
        for path in gradient_paths:
            gradient_stores.append(
                HiddenStateStore(
                    path,
                    layout[-1].end,
                    backend.hidden_size,
                    backend.teacher_dtype,
                )
            )

        _seed_hidden_store(backend, layout, hidden_stores[0])
        for layer_index in range(backend.num_layers):
            source = hidden_stores[layer_index]
            destination = hidden_stores[layer_index + 1]
            with backend.layer(layer_index, quantized=False) as layer:
                for item in layout:
                    hidden = source.read(
                        item.start,
                        item.end,
                        item.batch.input_ids.shape,
                        device=backend.device,
                    )
                    destination.write(
                        item.start,
                        backend.forward_layer(layer, layer_index, hidden),
                    )
            destination.flush()
            emit("fisher_forward", layer_index + 1, backend.num_layers, tokens=token_counts)

        final_norm = _load_final_norm(backend)
        head_weight = backend.index.tensor("lm_head.weight", device="cpu")
        generator = torch.Generator(device=backend.device)
        generator.manual_seed(seed + 17)
        try:
            for probe_index, item in enumerate(layout, start=1):
                hidden = hidden_stores[-1].read(
                    item.start,
                    item.end,
                    item.batch.input_ids.shape,
                    device=backend.device,
                )
                input_ids = torch.as_tensor(
                    item.batch.input_ids,
                    device=backend.device,
                    dtype=torch.int64,
                )
                gradient, probe_value = _chunked_head_gradient(
                    hidden,
                    input_ids,
                    final_norm,
                    head_weight,
                    generator,
                    row_chunk=head_row_chunk,
                )
                gradient_stores[0].write(item.start, gradient)
                collector.fisher_probes[item.split] += 1
                probe_sums[item.split] += probe_value
                emit("fisher_head", probe_index, len(layout), split=item.split)
            gradient_stores[0].flush()
        finally:
            del head_weight, final_norm
            gc.collect()
            if backend.device.type == "cuda":
                torch.cuda.empty_cache()

        collector.mode = "fisher"
        current_gradient, next_gradient = gradient_stores
        for layer_index in range(backend.num_layers - 1, -1, -1):
            with backend.layer(layer_index, quantized=False) as layer:
                collector.install_layer(layer, layer_index)
                try:
                    for item in layout:
                        collector.current_split = item.split
                        hidden = hidden_stores[layer_index].read(
                            item.start,
                            item.end,
                            item.batch.input_ids.shape,
                            device=backend.device,
                        )
                        hidden.requires_grad_(True)
                        upstream = current_gradient.read(
                            item.start,
                            item.end,
                            item.batch.input_ids.shape,
                            device=backend.device,
                        )
                        output = backend.forward_layer_with_grad(
                            layer,
                            layer_index,
                            hidden,
                        )
                        torch.autograd.backward(output, upstream)
                        if hidden.grad is None or not torch.isfinite(hidden.grad).all():
                            raise FloatingPointError(
                                f"non-finite hidden gradient at layer {layer_index}"
                            )
                        next_gradient.write(item.start, hidden.grad)
                finally:
                    collector.close()
            next_gradient.flush()
            current_gradient, next_gradient = next_gradient, current_gradient
            emit(
                "fisher_backward",
                backend.num_layers - layer_index,
                backend.num_layers,
                layer=layer_index,
            )
    finally:
        collector.mode = "idle"
        collector.close()
        backend.release_initial_state()
        for store in reversed(gradient_stores):
            store.close()
        for store in reversed(hidden_stores):
            store.close()
        if not keep_hidden:
            for path in all_paths:
                path.unlink(missing_ok=True)
    return token_counts, probe_sums


def collect_qwen35_statistics(
    model_path: str | Path,
    corpus: CalibrationCorpus,
    output: str | Path,
    *,
    device: str = "cuda:0",
    attention: str = "sdpa",
    input_window: int = 2048,
    fisher_window: int = 128,
    train_input_tokens: int = 1_572_864,
    validation_input_tokens: int = 262_144,
    train_fisher_tokens: int = 65_536,
    validation_fisher_tokens: int = 16_384,
    seed: int = 20260718,
    progress_every: int = 8,
    work_dir: str | Path | None = None,
    head_row_chunk: int = 0,
    keep_hidden: bool = False,
) -> CalibrationStatistics:
    """Collect Qwen3.5 statistics with one BF16 layer resident at a time."""

    try:
        from mfq.calibration.layerwise_qwen35 import Qwen35LayerwiseBackend
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Qwen3.5 calibration requires transformers") from exc
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("Qwen3.5 statistics collection requires CUDA")
    if attention not in {"eager", "sdpa"}:
        raise ValueError("attention must be eager or sdpa")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if head_row_chunk < 0:
        raise ValueError("head_row_chunk must be non-negative")

    root = Path(model_path).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"calibration statistics already exist: {output_path}")
    work = (
        Path(work_dir).resolve()
        if work_dir is not None
        else output_path.parent / f"{output_path.stem}.work"
    )
    work.mkdir(parents=True, exist_ok=True)
    targets = qwen35_linear_targets(root)
    cuda_device = torch.device(device)
    backend = Qwen35LayerwiseBackend(
        root,
        None,
        device=cuda_device,
        quant_backend="cpu",
        attention=attention,
    )
    collector = Qwen35StatisticsCollector(targets, cuda_device)
    head_rows = int(backend.index.shape("lm_head.weight")[0])
    resolved_head_chunk = head_rows if head_row_chunk == 0 else head_row_chunk
    head_gradient_mode = (
        "full_cross_entropy"
        if resolved_head_chunk >= head_rows
        else "chunked_softmax_bf16_accumulation"
    )
    torch.cuda.reset_peak_memory_stats(cuda_device)
    started = time.time()

    print(
        json.dumps(
            {
                "event": "calibration_contract",
                "model": str(root),
                "corpus": str(corpus.root),
                "output": str(output_path),
                "work_dir": str(work),
                "targets": len(targets),
                "input_tokens": {
                    "train": train_input_tokens,
                    "validation": validation_input_tokens,
                },
                "fisher_tokens": {
                    "train": train_fisher_tokens,
                    "validation": validation_fisher_tokens,
                },
                "input_window": input_window,
                "fisher_window": fisher_window,
                "attention": attention,
                "dtype": "bfloat16",
                "execution": "disk_backed_layer_streaming",
                "resident_weights": "embedding, one decoder layer, or lm_head",
                "head_row_chunk": resolved_head_chunk,
                "head_gradient_mode": head_gradient_mode,
                "graph": "HF Qwen3.5 text layers; no cache; original masks and layer order",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    def emit(phase: str, completed: int, total: int, **extra: Any) -> None:
        if completed % progress_every and completed != total:
            return
        print(
            json.dumps(
                {
                    "event": "progress",
                    "phase": phase,
                    "completed": completed,
                    "total": total,
                    **extra,
                    "seconds": round(time.time() - started, 3),
                    "cuda_allocated_gb": round(
                        torch.cuda.memory_allocated(cuda_device) / 1e9,
                        3,
                    ),
                    "cuda_peak_gb": round(
                        torch.cuda.max_memory_allocated(cuda_device) / 1e9,
                        3,
                    ),
                }
            ),
            flush=True,
        )

    input_counts = _collect_input_statistics(
        backend,
        collector,
        corpus,
        work,
        window_length=input_window,
        token_limits={
            "train": train_input_tokens,
            "validation": validation_input_tokens,
        },
        seed=seed,
        keep_hidden=keep_hidden,
        emit=emit,
    )
    fisher_counts, probe_sums = _collect_fisher_statistics(
        backend,
        collector,
        corpus,
        work,
        window_length=fisher_window,
        token_limits={
            "train": train_fisher_tokens,
            "validation": validation_fisher_tokens,
        },
        head_row_chunk=resolved_head_chunk,
        seed=seed,
        keep_hidden=keep_hidden,
        emit=emit,
    )
    entries = collector.entries()
    elapsed = time.time() - started
    metadata = {
        "objective": "kfac_diagonal_input_second_moment_x_rademacher_row_fisher",
        "fisher_loss": "next_token_cross_entropy",
        "model": _model_identity(root),
        "corpus": {
            "path": str(corpus.root),
            "manifest_sha256": _sha256(corpus.root / "manifest.json"),
            "manifest": corpus.manifest,
        },
        "device": device,
        "attention": attention,
        "dtype": "bfloat16",
        "seed": int(seed),
        "input_window": int(input_window),
        "fisher_window": int(fisher_window),
        "input_tokens": input_counts,
        "fisher_tokens": fisher_counts,
        "fisher_probe_sums": probe_sums,
        "target_count": len(targets),
        "execution": "disk_backed_layer_streaming",
        "head_row_chunk": int(resolved_head_chunk),
        "head_gradient_mode": head_gradient_mode,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(cuda_device)),
        "elapsed_seconds": elapsed,
    }
    save_statistics(output_path, entries, metadata)
    print(
        json.dumps(
            {
                "event": "statistics_saved",
                "output": str(output_path),
                "bytes": output_path.stat().st_size,
                "seconds": round(elapsed, 3),
                "cuda_peak_gb": round(
                    torch.cuda.max_memory_allocated(cuda_device) / 1e9,
                    3,
                ),
            }
        ),
        flush=True,
    )
    return load_statistics(output_path)


__all__ = [
    "CalibrationStatistics",
    "Qwen35StatisticsCollector",
    "TensorStatistics",
    "collect_qwen35_statistics",
    "load_statistics",
    "save_statistics",
]
