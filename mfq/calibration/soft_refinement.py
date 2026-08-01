"""Streamed end-to-end KL optimization of per-group precision probabilities."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from mfq.calibration.artifact import CalibrationScheme, load_scheme, save_scheme
from mfq.calibration.collector import HiddenStateStore, LayerwiseBackend
from mfq.calibration.dataset import CalibrationBatch, CalibrationCorpus
from mfq.calibration.evaluator import TensorCandidateEvaluation
from mfq.calibration.rate_distortion import (
    DiscreteSearchStep,
    PrecisionGroup,
    add_rate_gradient_,
    build_precision_groups,
    discretize_gate_logits,
    expected_storage_bits,
    fixed_storage_bits,
    initialize_gate_logits,
    refine_discrete_profiles,
    scheme_from_profiles,
    selected_storage_bits,
    update_dual,
)
from mfq.calibration.terminal_kl import ChunkedTerminalObjective
from mfq.formats.nint import NintSpec


class SoftRefinementBackend(LayerwiseBackend, Protocol):
    device: torch.device

    def layer_for_soft_assignment(
        self,
        layer_index: int,
        groups: Sequence[PrecisionGroup],
        logits: Mapping[str, torch.Tensor],
        temperature: float,
    ) -> AbstractContextManager[Any]: ...

    def layer_for_strategy(
        self,
        layer_index: int,
        specs: Mapping[str, NintSpec],
    ) -> AbstractContextManager[Any]: ...

    def forward_layer_with_grad(
        self,
        layer: Any,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor: ...

    def terminal_objective(self) -> AbstractContextManager[ChunkedTerminalObjective]: ...

    def clear_quantized_cache(self, layer_index: int | None = None) -> None: ...


SchemeBuilder = Callable[
    [
        CalibrationScheme,
        Sequence[PrecisionGroup],
        Mapping[str, str],
        Mapping[str, Any],
    ],
    CalibrationScheme,
]


@dataclass(frozen=True)
class SoftSearchStep:
    step: int
    temperature: float
    mean_kl: float
    expected_storage_bits: float
    expected_bpw: float
    dual: float
    rate_violation_percent: float
    gradient_norm: float
    epoch: int | None = None
    shard_index: int | None = None
    shard_tokens: int | None = None
    shard_scored_positions: int | None = None


@dataclass(frozen=True)
class SoftRefinementResult:
    scheme: CalibrationScheme
    steps: tuple[SoftSearchStep, ...]
    discrete_steps: tuple[DiscreteSearchStep, ...]
    base_train_kl: float
    selected_train_kl: float
    base_validation_kl: float
    selected_validation_kl: float
    accepted: bool


@dataclass(frozen=True)
class _BatchSlice:
    batch: CalibrationBatch
    start: int
    end: int


@dataclass
class _SplitWorkspace:
    name: str
    layout: tuple[_BatchSlice, ...]
    token_count: int
    scored_positions: int
    candidate_initial: HiddenStateStore
    teacher_buffers: tuple[HiddenStateStore, HiddenStateStore]
    teacher_final: HiddenStateStore | None
    hard_buffers: tuple[HiddenStateStore, ...]


@dataclass(frozen=True)
class _ShardSlice:
    batch: CalibrationBatch
    source_start: int
    source_end: int
    local_start: int
    local_end: int


@dataclass(frozen=True)
class _TrainingShard:
    index: int
    layout: tuple[_ShardSlice, ...]
    token_count: int
    scored_positions: int


def _layout(batches: Sequence[CalibrationBatch]) -> tuple[tuple[_BatchSlice, ...], int, int]:
    result: list[_BatchSlice] = []
    cursor = 0
    positions = 0
    for batch in batches:
        if batch.input_ids.ndim != 2 or batch.input_ids.shape[1] < 2:
            raise ValueError("rate-distortion batches require [batch, sequence>=2] token ids")
        count = int(batch.input_ids.size)
        result.append(_BatchSlice(batch, cursor, cursor + count))
        cursor += count
        positions += int(batch.input_ids.shape[0] * (batch.input_ids.shape[1] - 1))
    return tuple(result), cursor, positions


def _batch_digest(batches: Sequence[CalibrationBatch]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(str(tuple(int(value) for value in batch.input_ids.shape)).encode("ascii"))
        digest.update(batch.input_ids.tobytes(order="C"))
        digest.update(str(batch.chunk_indices).encode("ascii"))
    return digest.hexdigest()


def _precision_group_digest(groups: Sequence[PrecisionGroup]) -> str:
    payload = [
        {
            "name": group.name,
            "layer": group.layer,
            "tensor_names": group.tensor_names,
            "base_profile": group.base_profile,
            "options": [
                {
                    "profile": option.profile,
                    "storage_bits": option.storage_bits,
                    "specs": {
                        name: (spec.bits, spec.groupsize, spec.sub_bits)
                        for name, spec in sorted(option.specs.items())
                    },
                }
                for option in group.options
            ],
        }
        for group in groups
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _new_store(
    paths: list[Path],
    path: Path,
    token_count: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> HiddenStateStore:
    if path.exists():
        raise FileExistsError(f"rate-distortion hidden-state file already exists: {path}")
    paths.append(path)
    return HiddenStateStore(path, token_count, hidden_size, dtype)


def _create_split_workspace(
    backend: SoftRefinementBackend,
    work: Path,
    split: str,
    batches: Sequence[CalibrationBatch],
    paths: list[Path],
    *,
    create_hard_buffers: bool = True,
) -> _SplitWorkspace:
    layout, token_count, positions = _layout(batches)
    if not layout:
        raise ValueError(f"calibration corpus produced no {split} batches")
    created: list[HiddenStateStore] = []
    try:
        candidate_initial = _new_store(
            paths,
            work / f"soft-{split}-initial.bin",
            token_count,
            backend.hidden_size,
            backend.quantized_dtype,
        )
        created.append(candidate_initial)
        teacher_buffers = tuple(
            _new_store(
                paths,
                work / f"soft-{split}-teacher-{name}.bin",
                token_count,
                backend.hidden_size,
                backend.teacher_dtype,
            )
            for name in ("a", "b")
        )
        created.extend(teacher_buffers)
        hard_buffers: tuple[HiddenStateStore, ...] = ()
        if create_hard_buffers:
            hard_buffers = tuple(
                _new_store(
                    paths,
                    work / f"soft-{split}-hard-{name}.bin",
                    token_count,
                    backend.hidden_size,
                    backend.quantized_dtype,
                )
                for name in ("a", "b")
            )
            created.extend(hard_buffers)
        return _SplitWorkspace(
            split,
            layout,
            token_count,
            positions,
            candidate_initial,
            teacher_buffers,  # type: ignore[arg-type]
            None,
            hard_buffers,
        )
    except Exception:
        for store in reversed(created):
            store.close()
        raise


def _seed_and_run_teacher(
    backend: SoftRefinementBackend,
    workspaces: Sequence[_SplitWorkspace],
) -> None:
    for workspace in workspaces:
        for item in workspace.layout:
            ids = torch.as_tensor(
                item.batch.input_ids,
                device=backend.device,
                dtype=torch.int64,
            )
            hidden = backend.initial_hidden(ids)
            workspace.teacher_buffers[0].write(item.start, hidden)
            workspace.candidate_initial.write(
                item.start,
                hidden.to(dtype=backend.quantized_dtype),
            )
        workspace.teacher_buffers[0].flush()
        workspace.candidate_initial.flush()
    backend.release_initial_state()

    active = {workspace.name: 0 for workspace in workspaces}
    for layer_index in range(backend.num_layers):
        with backend.layer(layer_index, quantized=False) as layer:
            for workspace in workspaces:
                source_index = active[workspace.name]
                destination_index = 1 - source_index
                source = workspace.teacher_buffers[source_index]
                destination = workspace.teacher_buffers[destination_index]
                for item in workspace.layout:
                    shape = item.batch.input_ids.shape
                    hidden = source.read(
                        item.start,
                        item.end,
                        shape,
                        device=backend.device,
                    )
                    destination.write(
                        item.start,
                        backend.forward_layer(layer, layer_index, hidden),
                    )
                destination.flush()
                active[workspace.name] = destination_index
        print(
            json.dumps(
                {
                    "event": "soft_teacher_layer",
                    "layer": layer_index,
                    "layers": backend.num_layers,
                }
            ),
            flush=True,
        )
    for workspace in workspaces:
        workspace.teacher_final = workspace.teacher_buffers[active[workspace.name]]


def _release_stores(
    selected: Sequence[HiddenStateStore],
    stores: list[HiddenStateStore],
    paths: list[Path],
) -> None:
    for store in selected:
        path = store.path
        store.close()
        if store in stores:
            stores.remove(store)
        if path in paths:
            paths.remove(path)
        path.unlink(missing_ok=True)


def _release_teacher_scratch(
    workspace: _SplitWorkspace,
    stores: list[HiddenStateStore],
    paths: list[Path],
) -> None:
    if workspace.teacher_final is None:
        raise RuntimeError(f"teacher path for {workspace.name} was not initialized")
    scratch = [store for store in workspace.teacher_buffers if store is not workspace.teacher_final]
    _release_stores(scratch, stores, paths)


def _partition_training_shards(
    layout: Sequence[_BatchSlice],
    shard_tokens: int,
) -> tuple[_TrainingShard, ...]:
    if shard_tokens <= 0:
        raise ValueError("soft shard token count must be positive")
    shards: list[_TrainingShard] = []
    current: list[_ShardSlice] = []
    tokens = 0
    positions = 0

    def finish() -> None:
        nonlocal current, tokens, positions
        if not current:
            return
        shards.append(_TrainingShard(len(shards), tuple(current), tokens, positions))
        current = []
        tokens = 0
        positions = 0

    for item in layout:
        count = item.end - item.start
        if count > shard_tokens:
            raise ValueError(
                f"calibration batch has {count} tokens, exceeding soft shard size "
                f"{shard_tokens}; reduce batch size or increase --soft-shard-tokens"
            )
        if current and tokens + count > shard_tokens:
            finish()
        current.append(
            _ShardSlice(
                item.batch,
                item.start,
                item.end,
                tokens,
                tokens + count,
            )
        )
        tokens += count
        positions += int(item.batch.input_ids.shape[0] * (item.batch.input_ids.shape[1] - 1))
    finish()
    if not shards:
        raise ValueError("calibration corpus produced no soft training shards")
    return tuple(shards)


def _soft_forward_shard(
    backend: SoftRefinementBackend,
    workspace: _SplitWorkspace,
    shard: _TrainingShard,
    hidden_stores: Sequence[HiddenStateStore],
    groups: Sequence[PrecisionGroup],
    logits: Mapping[str, torch.Tensor],
    temperature: float,
) -> None:
    if len(hidden_stores) != backend.num_layers:
        raise ValueError("soft shard must provide one hidden store per decoder layer")
    soft_dtype = getattr(backend, "soft_dtype", backend.quantized_dtype)
    for layer_index in range(backend.num_layers):
        with backend.layer_for_soft_assignment(
            layer_index,
            groups,
            logits,
            temperature,
        ) as layer:
            for item in shard.layout:
                shape = item.batch.input_ids.shape
                source = (
                    workspace.candidate_initial
                    if layer_index == 0
                    else hidden_stores[layer_index - 1]
                )
                start = item.source_start if layer_index == 0 else item.local_start
                end = item.source_end if layer_index == 0 else item.local_end
                hidden = source.read(start, end, shape, device=backend.device).to(dtype=soft_dtype)
                hidden_stores[layer_index].write(
                    item.local_start,
                    backend.forward_layer(layer, layer_index, hidden),
                )
        hidden_stores[layer_index].flush()


def _terminal_metric_shard(
    backend: SoftRefinementBackend,
    objective: ChunkedTerminalObjective,
    workspace: _SplitWorkspace,
    shard: _TrainingShard,
    candidate: HiddenStateStore,
    *,
    row_chunk: int,
    gradient_store: HiddenStateStore,
) -> float:
    if workspace.teacher_final is None:
        raise RuntimeError(f"teacher path for {workspace.name} was not initialized")
    total = 0.0
    for item in shard.layout:
        shape = item.batch.input_ids.shape
        reference = workspace.teacher_final.read(
            item.source_start,
            item.source_end,
            shape,
            device=backend.device,
        )
        value = candidate.read(
            item.local_start,
            item.local_end,
            shape,
            device=backend.device,
        )
        result = objective.evaluate(
            reference,
            value,
            row_chunk=row_chunk,
            with_gradient=True,
            gradient_scale=1.0 / shard.scored_positions,
        )
        if result.gradient is None:
            raise RuntimeError("terminal KL omitted its requested hidden gradient")
        total += result.sum_kl
        gradient_store.write(item.local_start, result.gradient)
    gradient_store.flush()
    return total / shard.scored_positions


def _soft_backward_shard(
    backend: SoftRefinementBackend,
    workspace: _SplitWorkspace,
    shard: _TrainingShard,
    hidden_stores: Sequence[HiddenStateStore],
    gradient_stores: tuple[HiddenStateStore, HiddenStateStore],
    groups: Sequence[PrecisionGroup],
    logits: Mapping[str, torch.Tensor],
    temperature: float,
) -> None:
    soft_dtype = getattr(backend, "soft_dtype", backend.quantized_dtype)
    current, destination = gradient_stores
    for layer_index in range(backend.num_layers - 1, -1, -1):
        with backend.layer_for_soft_assignment(
            layer_index,
            groups,
            logits,
            temperature,
        ) as layer:
            for item in shard.layout:
                shape = item.batch.input_ids.shape
                source = (
                    workspace.candidate_initial
                    if layer_index == 0
                    else hidden_stores[layer_index - 1]
                )
                start = item.source_start if layer_index == 0 else item.local_start
                end = item.source_end if layer_index == 0 else item.local_end
                hidden = source.read(start, end, shape, device=backend.device).to(dtype=soft_dtype)
                hidden.requires_grad_(True)
                upstream = current.read(
                    item.local_start,
                    item.local_end,
                    shape,
                    device=backend.device,
                )
                output = backend.forward_layer_with_grad(
                    layer,
                    layer_index,
                    hidden,
                )
                torch.autograd.backward(output, upstream.to(dtype=output.dtype))
                if hidden.grad is None or not torch.isfinite(hidden.grad).all():
                    raise FloatingPointError(
                        f"non-finite soft hidden gradient at layer {layer_index}, "
                        f"shard {shard.index}"
                    )
                destination.write(item.local_start, hidden.grad)
        destination.flush()
        current, destination = destination, current


def _soft_forward(
    backend: SoftRefinementBackend,
    workspace: _SplitWorkspace,
    stores: Sequence[HiddenStateStore],
    groups: Sequence[PrecisionGroup],
    logits: Mapping[str, torch.Tensor],
    temperature: float,
) -> None:
    soft_dtype = getattr(backend, "soft_dtype", backend.quantized_dtype)
    for layer_index in range(backend.num_layers):
        with backend.layer_for_soft_assignment(
            layer_index,
            groups,
            logits,
            temperature,
        ) as layer:
            for item in workspace.layout:
                shape = item.batch.input_ids.shape
                hidden = (
                    stores[layer_index]
                    .read(
                        item.start,
                        item.end,
                        shape,
                        device=backend.device,
                    )
                    .to(dtype=soft_dtype)
                )
                stores[layer_index + 1].write(
                    item.start,
                    backend.forward_layer(layer, layer_index, hidden),
                )
        stores[layer_index + 1].flush()


def _terminal_metric(
    backend: SoftRefinementBackend,
    objective: ChunkedTerminalObjective,
    workspace: _SplitWorkspace,
    candidate: HiddenStateStore,
    *,
    row_chunk: int,
    gradient_store: HiddenStateStore | None = None,
) -> float:
    if workspace.teacher_final is None:
        raise RuntimeError(f"teacher path for {workspace.name} was not initialized")
    total = 0.0
    with_gradient = gradient_store is not None
    for item in workspace.layout:
        shape = item.batch.input_ids.shape
        reference = workspace.teacher_final.read(
            item.start,
            item.end,
            shape,
            device=backend.device,
        )
        value = candidate.read(
            item.start,
            item.end,
            shape,
            device=backend.device,
        )
        result = objective.evaluate(
            reference,
            value,
            row_chunk=row_chunk,
            with_gradient=with_gradient,
            gradient_scale=1.0 / workspace.scored_positions,
        )
        total += result.sum_kl
        if gradient_store is not None:
            if result.gradient is None:
                raise RuntimeError("terminal KL omitted its requested hidden gradient")
            gradient_store.write(item.start, result.gradient)
    if gradient_store is not None:
        gradient_store.flush()
    return total / workspace.scored_positions


def _soft_backward(
    backend: SoftRefinementBackend,
    workspace: _SplitWorkspace,
    stores: Sequence[HiddenStateStore],
    gradient_stores: tuple[HiddenStateStore, HiddenStateStore],
    groups: Sequence[PrecisionGroup],
    logits: Mapping[str, torch.Tensor],
    temperature: float,
) -> None:
    soft_dtype = getattr(backend, "soft_dtype", backend.quantized_dtype)
    current, destination = gradient_stores
    for layer_index in range(backend.num_layers - 1, -1, -1):
        with backend.layer_for_soft_assignment(
            layer_index,
            groups,
            logits,
            temperature,
        ) as layer:
            for item in workspace.layout:
                shape = item.batch.input_ids.shape
                hidden = (
                    stores[layer_index]
                    .read(
                        item.start,
                        item.end,
                        shape,
                        device=backend.device,
                    )
                    .to(dtype=soft_dtype)
                )
                hidden.requires_grad_(True)
                upstream = current.read(
                    item.start,
                    item.end,
                    shape,
                    device=backend.device,
                )
                output = backend.forward_layer_with_grad(
                    layer,
                    layer_index,
                    hidden,
                )
                torch.autograd.backward(output, upstream.to(dtype=output.dtype))
                if hidden.grad is None or not torch.isfinite(hidden.grad).all():
                    raise FloatingPointError(
                        f"non-finite soft hidden gradient at layer {layer_index}"
                    )
                destination.write(item.start, hidden.grad)
        destination.flush()
        current, destination = destination, current


def _specs_by_layer(
    groups: Sequence[PrecisionGroup],
    profiles: Mapping[str, str],
) -> dict[int, dict[str, NintSpec]]:
    result: dict[int, dict[str, NintSpec]] = defaultdict(dict)
    for group in groups:
        option = group.require(profiles[group.name])
        overlap = set(result[group.layer]).intersection(option.specs)
        if overlap:
            raise ValueError(f"layer {group.layer} repeats tensors: {sorted(overlap)[:4]}")
        result[group.layer].update(option.specs)
    return dict(result)


def _hard_metric(
    backend: SoftRefinementBackend,
    objective: ChunkedTerminalObjective,
    workspace: _SplitWorkspace,
    groups: Sequence[PrecisionGroup],
    profiles: Mapping[str, str],
    *,
    row_chunk: int,
) -> float:
    if len(workspace.hard_buffers) != 2:
        raise RuntimeError(f"hard buffers for {workspace.name} were not allocated")
    specs_by_layer = _specs_by_layer(groups, profiles)
    source = workspace.candidate_initial
    for layer_index in range(backend.num_layers):
        destination = workspace.hard_buffers[layer_index % 2]
        try:
            with backend.layer_for_strategy(
                layer_index,
                specs_by_layer[layer_index],
            ) as layer:
                for item in workspace.layout:
                    shape = item.batch.input_ids.shape
                    hidden = source.read(
                        item.start,
                        item.end,
                        shape,
                        device=backend.device,
                    )
                    destination.write(
                        item.start,
                        backend.forward_layer(layer, layer_index, hidden),
                    )
            destination.flush()
        finally:
            backend.clear_quantized_cache(layer_index)
        source = destination
    return _terminal_metric(
        backend,
        objective,
        workspace,
        source,
        row_chunk=row_chunk,
    )


def _temperature(step: int, steps: int, start: float, end: float) -> float:
    if start <= 0 or end <= 0:
        raise ValueError("soft-search temperatures must be positive")
    if steps == 1:
        return end
    ratio = step / (steps - 1)
    return start * (end / start) ** ratio


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    dual: float,
    logits: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    groups: Sequence[PrecisionGroup],
    history: Sequence[SoftSearchStep],
    schedule: Mapping[str, Any] | None = None,
) -> None:
    document = {
        "format": "mfq.soft-rate-distortion-checkpoint.v1",
        "step": int(step),
        "dual": float(dual),
        "profiles": {group.name: [option.profile for option in group.options] for group in groups},
        "logits": {name: value.detach().cpu() for name, value in logits.items()},
        "optimizer": optimizer.state_dict(),
        "history": [asdict(item) for item in history],
        "schedule": None if schedule is None else dict(schedule),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(document, temporary)
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path,
    groups: Sequence[PrecisionGroup],
    logits: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    expected_schedule: Mapping[str, Any] | None = None,
) -> tuple[int, float, list[SoftSearchStep]]:
    document = torch.load(path, map_location="cpu", weights_only=False)
    if document.get("format") != "mfq.soft-rate-distortion-checkpoint.v1":
        raise ValueError(f"unsupported soft-search checkpoint: {path}")
    expected = {group.name: [option.profile for option in group.options] for group in groups}
    if document.get("profiles") != expected:
        raise ValueError("soft-search checkpoint precision groups changed")
    if set(document["logits"]) != set(logits):
        raise ValueError("soft-search checkpoint logits do not cover every group")
    if expected_schedule is not None and document.get("schedule") != dict(expected_schedule):
        raise ValueError("soft-search checkpoint data/update schedule changed")
    for name, value in logits.items():
        value.data.copy_(document["logits"][name].to(device=value.device))
    optimizer.load_state_dict(document["optimizer"])
    history = [SoftSearchStep(**item) for item in document.get("history", [])]
    if history and history[-1].step != int(document["step"]):
        raise ValueError("soft-search checkpoint history does not match its step")
    return int(document["step"]), float(document["dual"]), history


def _refine_soft_rate_distortion_sharded(
    backend: SoftRefinementBackend,
    corpus: CalibrationCorpus,
    validation_corpus: CalibrationCorpus,
    base_scheme: CalibrationScheme,
    groups: Sequence[PrecisionGroup],
    evaluations: Mapping[str, Mapping[str, TensorCandidateEvaluation]],
    scheme_builder: SchemeBuilder,
    output_scheme: str | Path,
    report: str | Path,
    *,
    work_dir: str | Path,
    window_length: int,
    batch_size: int,
    train_tokens: int,
    validation_tokens: int,
    hard_train_tokens: int,
    seed: int,
    shard_tokens: int,
    epochs: int,
    learning_rate: float,
    dual_learning_rate: float,
    initial_dual: float,
    temperature_start: float,
    temperature_end: float,
    head_row_chunk: int,
    discrete_iterations: int,
    discrete_single_candidates: int,
    discrete_pair_candidates: int,
    validation_tolerance: float,
    resume: bool,
    keep_hidden: bool,
) -> SoftRefinementResult:
    if min(train_tokens, validation_tokens, hard_train_tokens, shard_tokens, epochs) <= 0:
        raise ValueError("sharded soft-search token counts and epochs must be positive")
    if learning_rate <= 0 or dual_learning_rate < 0 or initial_dual < 0:
        raise ValueError("soft-search learning rates and initial dual are invalid")
    if discrete_iterations < 0:
        raise ValueError("discrete_iterations must be non-negative")
    if validation_tolerance < 0:
        raise ValueError("validation_tolerance must be non-negative")

    scheme_path = Path(output_scheme).resolve()
    report_path = Path(report).resolve()
    if scheme_path.exists() or report_path.exists():
        raise FileExistsError("soft rate-distortion output already exists")
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    session_work = work / f"soft-session-{os.getpid()}"
    if session_work.exists():
        raise FileExistsError(f"soft-search session directory already exists: {session_work}")
    checkpoint_path = work / "soft-rate-distortion-checkpoint.pt"
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(f"soft-search checkpoint already exists: {checkpoint_path}")
    if resume and not checkpoint_path.is_file():
        raise FileNotFoundError(f"soft-search checkpoint does not exist: {checkpoint_path}")

    groups = tuple(groups)
    if {group.layer for group in groups} != set(range(backend.num_layers)):
        raise ValueError("soft precision groups do not cover every decoder layer")
    fixed_bits = fixed_storage_bits(base_scheme, groups)

    train_batches = list(
        corpus.iter_batches(
            "train",
            window_length=window_length,
            batch_size=batch_size,
            max_tokens=train_tokens,
            seed=seed,
        )
    )
    train_layout, train_token_count, _train_positions = _layout(train_batches)
    shards = _partition_training_shards(train_layout, shard_tokens)
    total_updates = len(shards) * epochs
    hard_train_batches = list(
        corpus.iter_batches(
            "train",
            window_length=window_length,
            batch_size=batch_size,
            max_tokens=min(hard_train_tokens, train_token_count),
            seed=seed,
        )
    )
    validation_batches = list(
        validation_corpus.iter_batches(
            "validation",
            window_length=window_length,
            batch_size=batch_size,
            max_tokens=validation_tokens,
            seed=seed + 1,
        )
    )
    hard_train_token_count = sum(int(batch.input_ids.size) for batch in hard_train_batches)
    validation_token_count = sum(int(batch.input_ids.size) for batch in validation_batches)
    minimum_rows = int(getattr(backend, "soft_min_autograd_rows", 0))
    if minimum_rows and any(batch.input_ids.size < minimum_rows for batch in train_batches):
        raise ValueError(
            f"soft-search train batches must contain at least {minimum_rows} rows "
            "so the differentiable dequant+GEMM path is used"
        )

    schedule = {
        "mode": "sharded",
        "model": str(getattr(backend, "root", "custom-backend")),
        "corpus": str(corpus.root),
        "validation_corpus": str(validation_corpus.root),
        "base_scheme": str(base_scheme.path),
        "target_storage_bits": int(base_scheme.target_storage_bits),
        "precision_group_digest": _precision_group_digest(groups),
        "train_digest": _batch_digest(train_batches),
        "hard_train_digest": _batch_digest(hard_train_batches),
        "validation_digest": _batch_digest(validation_batches),
        "window_length": int(window_length),
        "batch_size": int(batch_size),
        "train_tokens": int(train_token_count),
        "hard_train_tokens": int(hard_train_token_count),
        "validation_tokens": int(validation_token_count),
        "shard_tokens": int(shard_tokens),
        "shard_count": len(shards),
        "epochs": int(epochs),
        "total_updates": int(total_updates),
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "dual_learning_rate": float(dual_learning_rate),
        "initial_dual": float(initial_dual),
        "temperature_start": float(temperature_start),
        "temperature_end": float(temperature_end),
        "head_row_chunk": int(head_row_chunk),
        "discrete_iterations": int(discrete_iterations),
        "discrete_single_candidates": int(discrete_single_candidates),
        "discrete_pair_candidates": int(discrete_pair_candidates),
        "validation_tolerance": float(validation_tolerance),
    }
    logits = initialize_gate_logits(groups, device=backend.device)
    optimizer = torch.optim.Adam(list(logits.values()), lr=learning_rate)
    start_update = 0
    dual = float(initial_dual)
    steps_report: list[SoftSearchStep] = []
    if resume:
        completed, dual, steps_report = _load_checkpoint(
            checkpoint_path,
            groups,
            logits,
            optimizer,
            expected_schedule=schedule,
        )
        start_update = completed + 1
    if start_update > total_updates:
        raise ValueError("soft-search checkpoint exceeds the requested shard schedule")

    teacher_bytes = torch.empty((), dtype=backend.teacher_dtype).element_size()
    candidate_bytes = torch.empty((), dtype=backend.quantized_dtype).element_size()
    soft_dtype = getattr(backend, "soft_dtype", backend.quantized_dtype)
    soft_bytes = torch.empty((), dtype=soft_dtype).element_size()
    max_shard_tokens = max(shard.token_count for shard in shards)
    teacher_peak_bytes = (
        train_token_count * backend.hidden_size * (candidate_bytes + 2 * teacher_bytes)
    )
    persistent_train_bytes = (
        train_token_count * backend.hidden_size * (candidate_bytes + teacher_bytes)
    )
    shard_workspace_bytes = (
        max_shard_tokens * backend.hidden_size * (backend.num_layers * soft_bytes + 8)
    )
    hard_workspace_bytes = (hard_train_token_count + validation_token_count) * (
        backend.hidden_size * (3 * candidate_bytes + 2 * teacher_bytes)
    )
    hidden_disk_bytes = max(
        teacher_peak_bytes,
        persistent_train_bytes + shard_workspace_bytes,
        hard_workspace_bytes,
    )
    packed_candidate_bytes = (
        sum(option.storage_bits for group in groups for option in group.options) // 8
    )
    print(
        json.dumps(
            {
                "event": "soft_search_contract",
                "mode": "sharded",
                "model": str(getattr(backend, "root", "custom-backend")),
                "corpus": str(corpus.root),
                "validation_corpus": str(validation_corpus.root),
                "window_length": int(window_length),
                "batch_size": int(batch_size),
                "train_tokens": train_token_count,
                "hard_train_tokens": hard_train_token_count,
                "validation_tokens": validation_token_count,
                "shard_tokens": int(shard_tokens),
                "max_actual_shard_tokens": int(max_shard_tokens),
                "shard_count": len(shards),
                "epochs": int(epochs),
                "updates": int(total_updates),
                "training_tokens_seen": int(train_token_count * epochs),
                "objective": "mean_KL(BF16_teacher||relaxed_MFQ)",
                "soft_compute_dtype": str(soft_dtype).removeprefix("torch."),
                "hard_compute_dtype": str(backend.quantized_dtype).removeprefix("torch."),
                "group_count": len(groups),
                "candidate_count": sum(len(group.options) for group in groups),
                "target_storage_bits": int(base_scheme.target_storage_bits),
                "estimated_peak_hidden_disk_bytes": int(hidden_disk_bytes),
                "estimated_packed_candidate_bytes": int(packed_candidate_bytes),
                "output_scheme": str(scheme_path),
                "report": str(report_path),
                "checkpoint": str(checkpoint_path),
            }
        ),
        flush=True,
    )

    hidden_paths: list[Path] = []
    stores: list[HiddenStateStore] = []
    discrete_steps: tuple[DiscreteSearchStep, ...] = ()
    result: SoftRefinementResult | None = None
    try:
        session_work.mkdir()
        if start_update < total_updates:
            train = _create_split_workspace(
                backend,
                session_work,
                "train-all",
                train_batches,
                hidden_paths,
                create_hard_buffers=False,
            )
            stores.extend([train.candidate_initial, *train.teacher_buffers])
            _seed_and_run_teacher(backend, (train,))
            _release_teacher_scratch(train, stores, hidden_paths)

            shard_hidden: list[HiddenStateStore] = []
            for layer_index in range(backend.num_layers):
                store = _new_store(
                    hidden_paths,
                    session_work / f"soft-shard-hidden-{layer_index:03d}.bin",
                    max_shard_tokens,
                    backend.hidden_size,
                    soft_dtype,
                )
                shard_hidden.append(store)
                stores.append(store)
            gradient_store_list: list[HiddenStateStore] = []
            for name in ("a", "b"):
                store = _new_store(
                    hidden_paths,
                    session_work / f"soft-shard-gradient-{name}.bin",
                    max_shard_tokens,
                    backend.hidden_size,
                    torch.float32,
                )
                gradient_store_list.append(store)
                stores.append(store)
            gradient_stores = (gradient_store_list[0], gradient_store_list[1])

            with backend.terminal_objective() as objective:
                for update in range(start_update, total_updates):
                    epoch = update // len(shards)
                    shard = shards[update % len(shards)]
                    temperature = _temperature(
                        update,
                        total_updates,
                        temperature_start,
                        temperature_end,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    _soft_forward_shard(
                        backend,
                        train,
                        shard,
                        shard_hidden,
                        groups,
                        logits,
                        temperature,
                    )
                    mean_kl = _terminal_metric_shard(
                        backend,
                        objective,
                        train,
                        shard,
                        shard_hidden[-1],
                        row_chunk=head_row_chunk,
                        gradient_store=gradient_stores[0],
                    )
                    _soft_backward_shard(
                        backend,
                        train,
                        shard,
                        shard_hidden,
                        gradient_stores,
                        groups,
                        logits,
                        temperature,
                    )
                    expected_bits = float(
                        expected_storage_bits(
                            groups,
                            logits,
                            temperature=temperature,
                            fixed_bits=fixed_bits,
                        )
                        .detach()
                        .item()
                    )
                    add_rate_gradient_(
                        groups,
                        logits,
                        temperature=temperature,
                        target_storage_bits=base_scheme.target_storage_bits,
                        dual=dual,
                    )
                    gradient_norm = float(
                        torch.nn.utils.clip_grad_norm_(list(logits.values()), max_norm=5.0).item()
                    )
                    optimizer.step()
                    dual = update_dual(
                        dual,
                        expected_bits,
                        base_scheme.target_storage_bits,
                        learning_rate=dual_learning_rate,
                    )
                    item = SoftSearchStep(
                        step=update,
                        temperature=temperature,
                        mean_kl=mean_kl,
                        expected_storage_bits=expected_bits,
                        expected_bpw=expected_bits / base_scheme.weight_count,
                        dual=dual,
                        rate_violation_percent=(
                            100.0
                            * (expected_bits - base_scheme.target_storage_bits)
                            / base_scheme.target_storage_bits
                        ),
                        gradient_norm=gradient_norm,
                        epoch=epoch,
                        shard_index=shard.index,
                        shard_tokens=shard.token_count,
                        shard_scored_positions=shard.scored_positions,
                    )
                    steps_report.append(item)
                    print(
                        json.dumps({"event": "soft_search_step", **asdict(item)}),
                        flush=True,
                    )
                    _save_checkpoint(
                        checkpoint_path,
                        step=update,
                        dual=dual,
                        logits=logits,
                        optimizer=optimizer,
                        groups=groups,
                        history=steps_report,
                        schedule=schedule,
                    )

            _release_stores(tuple(stores), stores, hidden_paths)
            backend.clear_quantized_cache()

        hard_train = _create_split_workspace(
            backend,
            session_work,
            "hard-train",
            hard_train_batches,
            hidden_paths,
        )
        stores.extend(
            [hard_train.candidate_initial, *hard_train.teacher_buffers, *hard_train.hard_buffers]
        )
        validation = _create_split_workspace(
            backend,
            session_work,
            "validation",
            validation_batches,
            hidden_paths,
        )
        stores.extend(
            [validation.candidate_initial, *validation.teacher_buffers, *validation.hard_buffers]
        )
        _seed_and_run_teacher(backend, (hard_train, validation))

        with backend.terminal_objective() as objective:
            initial_profiles = discretize_gate_logits(
                groups,
                logits,
                target_storage_bits=base_scheme.target_storage_bits,
                fixed_bits=fixed_bits,
            )
            base_profiles = {group.name: group.base_profile for group in groups}
            train_metric_cache: dict[tuple[tuple[str, str], ...], float] = {}

            def train_metric(profiles: Mapping[str, str]) -> float:
                key = tuple(sorted(profiles.items()))
                if key not in train_metric_cache:
                    metric = _hard_metric(
                        backend,
                        objective,
                        hard_train,
                        groups,
                        profiles,
                        row_chunk=head_row_chunk,
                    )
                    train_metric_cache[key] = metric
                    print(
                        json.dumps(
                            {
                                "event": "soft_hard_train_candidate",
                                "candidate_index": len(train_metric_cache),
                                "mean_kl": metric,
                                "storage_bits": selected_storage_bits(
                                    groups, profiles, fixed_bits=fixed_bits
                                ),
                                "changed_from_initial": sum(
                                    profiles[group.name] != initial_profiles[group.name]
                                    for group in groups
                                ),
                            }
                        ),
                        flush=True,
                    )
                return train_metric_cache[key]

            if discrete_iterations == 0:
                initial_metric = train_metric(initial_profiles)
                refined_profiles = dict(initial_profiles)
                discrete_steps = (
                    DiscreteSearchStep(
                        iteration=0,
                        metric=initial_metric,
                        storage_bits=selected_storage_bits(
                            groups, initial_profiles, fixed_bits=fixed_bits
                        ),
                        changed_groups=(),
                    ),
                )
            else:
                refined_profiles, discrete_steps = refine_discrete_profiles(
                    groups,
                    initial_profiles,
                    logits,
                    train_metric,
                    target_storage_bits=base_scheme.target_storage_bits,
                    fixed_bits=fixed_bits,
                    max_iterations=discrete_iterations,
                    max_single=discrete_single_candidates,
                    max_pair=discrete_pair_candidates,
                )
            base_train_kl = train_metric(base_profiles)
            refined_train_kl = train_metric(refined_profiles)
            base_validation_kl = _hard_metric(
                backend,
                objective,
                validation,
                groups,
                base_profiles,
                row_chunk=head_row_chunk,
            )
            print(
                json.dumps(
                    {
                        "event": "soft_hard_validation",
                        "candidate": "base",
                        "mean_kl": base_validation_kl,
                    }
                ),
                flush=True,
            )
            refined_validation_kl = _hard_metric(
                backend,
                objective,
                validation,
                groups,
                refined_profiles,
                row_chunk=head_row_chunk,
            )
            print(
                json.dumps(
                    {
                        "event": "soft_hard_validation",
                        "candidate": "refined",
                        "mean_kl": refined_validation_kl,
                    }
                ),
                flush=True,
            )
            train_tolerance = max(1e-12, base_train_kl * validation_tolerance)
            validation_limit = max(1e-12, base_validation_kl * validation_tolerance)
            accepted = (
                refined_train_kl <= base_train_kl + train_tolerance
                and refined_validation_kl <= base_validation_kl + validation_limit
            )
            selected_profiles = refined_profiles if accepted else base_profiles
            selected_train_kl = refined_train_kl if accepted else base_train_kl
            selected_validation_kl = refined_validation_kl if accepted else base_validation_kl

            metadata = {
                "objective": "mean_KL(BF16_teacher||relaxed_MFQ)",
                "mode": "sharded",
                "soft_compute_dtype": str(soft_dtype).removeprefix("torch."),
                "hard_compute_dtype": str(backend.quantized_dtype).removeprefix("torch."),
                "decision_unit": "per_layer_runtime_precision_group",
                "train_corpus": str(corpus.root),
                "validation_corpus": str(validation_corpus.root),
                "window_length": int(window_length),
                "batch_size": int(batch_size),
                "train_tokens": int(train_token_count),
                "hard_train_tokens": int(hard_train.token_count),
                "validation_tokens": int(validation.token_count),
                "shard_tokens": int(shard_tokens),
                "shard_count": len(shards),
                "epochs": int(epochs),
                "updates": int(total_updates),
                "learning_rate": float(learning_rate),
                "dual_learning_rate": float(dual_learning_rate),
                "temperature_start": float(temperature_start),
                "temperature_end": float(temperature_end),
                "head_row_chunk": int(head_row_chunk),
                "group_count": len(groups),
                "accepted_on_validation": accepted,
                "base_train_kl": base_train_kl,
                "selected_train_kl": selected_train_kl,
                "base_validation_kl": base_validation_kl,
                "selected_validation_kl": selected_validation_kl,
            }
            scheme = scheme_builder(
                base_scheme,
                groups,
                selected_profiles,
                {"soft_rate_distortion": metadata},
            )
            save_scheme(scheme_path, scheme)
            loaded = load_scheme(scheme_path)
            report_document = {
                "format": "mfq.soft-rate-distortion.v1",
                "base_scheme": str(base_scheme.path),
                "output_scheme": str(scheme_path),
                "corpus": str(corpus.root),
                "validation_corpus": str(validation_corpus.root),
                "mode": "sharded",
                "target_storage_bits": int(base_scheme.target_storage_bits),
                "actual_storage_bits": int(loaded.storage_bits),
                "actual_bpw": loaded.bpw,
                "fixed_storage_bits": int(fixed_bits),
                "group_count": len(groups),
                "candidate_count": sum(len(group.options) for group in groups),
                "soft_compute_dtype": str(soft_dtype).removeprefix("torch."),
                "hard_compute_dtype": str(backend.quantized_dtype).removeprefix("torch."),
                "train_tokens": int(train_token_count),
                "hard_train_tokens": int(hard_train.token_count),
                "validation_tokens": int(validation.token_count),
                "shard_tokens": int(shard_tokens),
                "shard_count": len(shards),
                "epochs": int(epochs),
                "updates": int(total_updates),
                "accepted_on_validation": accepted,
                "base_train_kl": base_train_kl,
                "refined_train_kl": refined_train_kl,
                "selected_train_kl": selected_train_kl,
                "base_validation_kl": base_validation_kl,
                "refined_validation_kl": refined_validation_kl,
                "selected_validation_kl": selected_validation_kl,
                "initial_train_kl": discrete_steps[0].metric,
                "steps": [asdict(item) for item in steps_report],
                "discrete_steps": [asdict(item) for item in discrete_steps],
                "initial_profiles": dict(sorted(initial_profiles.items())),
                "refined_profiles": dict(sorted(refined_profiles.items())),
                "selected_profiles": dict(sorted(selected_profiles.items())),
                "learned_logits": {
                    group.name: {
                        option.profile: float(logits[group.name][index].detach().cpu().item())
                        for index, option in enumerate(group.options)
                    }
                    for group in groups
                },
                "selected_storage_bits": selected_storage_bits(
                    groups, selected_profiles, fixed_bits=fixed_bits
                ),
                "initial_storage_bits": selected_storage_bits(
                    groups, initial_profiles, fixed_bits=fixed_bits
                ),
                "refined_storage_bits": selected_storage_bits(
                    groups, refined_profiles, fixed_bits=fixed_bits
                ),
            }
            report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = report_path.with_suffix(report_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(report_document, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, report_path)
            result = SoftRefinementResult(
                scheme=loaded,
                steps=tuple(steps_report),
                discrete_steps=discrete_steps,
                base_train_kl=base_train_kl,
                selected_train_kl=selected_train_kl,
                base_validation_kl=base_validation_kl,
                selected_validation_kl=selected_validation_kl,
                accepted=accepted,
            )
    finally:
        backend.release_initial_state()
        backend.clear_quantized_cache()
        for store in reversed(stores):
            store.close()
        if not keep_hidden:
            for path in hidden_paths:
                path.unlink(missing_ok=True)
            if session_work.exists():
                session_work.rmdir()
    if result is None:
        raise RuntimeError("sharded soft rate-distortion refinement produced no result")
    return result


def refine_soft_rate_distortion(
    backend: SoftRefinementBackend,
    corpus: CalibrationCorpus,
    base_scheme: CalibrationScheme,
    evaluations: Mapping[str, Mapping[str, TensorCandidateEvaluation]],
    output_scheme: str | Path,
    report: str | Path,
    *,
    work_dir: str | Path,
    window_length: int = 256,
    batch_size: int = 8,
    train_tokens: int = 8_192,
    validation_tokens: int = 4_096,
    validation_corpus: CalibrationCorpus | None = None,
    shard_tokens: int = 0,
    epochs: int = 1,
    hard_train_tokens: int = 262_144,
    seed: int = 20260718,
    steps: int = 20,
    learning_rate: float = 0.08,
    dual_learning_rate: float = 0.1,
    initial_dual: float = 0.0,
    temperature_start: float = 1.5,
    temperature_end: float = 0.25,
    head_row_chunk: int = 2_048,
    discrete_iterations: int = 2,
    discrete_single_candidates: int = 8,
    discrete_pair_candidates: int = 16,
    validation_tolerance: float = 0.0,
    resume: bool = False,
    keep_hidden: bool = False,
    precision_groups: Sequence[PrecisionGroup] | None = None,
    scheme_builder: SchemeBuilder | None = None,
) -> SoftRefinementResult:
    """Optimize every runtime precision group against end-to-end teacher KL."""

    if precision_groups is None:
        groups = build_precision_groups(evaluations, base_scheme)
    else:
        groups = tuple(precision_groups)
        if not groups:
            raise ValueError("soft refinement precision_groups cannot be empty")
    if scheme_builder is None:
        def build_output_scheme(
            scheme: CalibrationScheme,
            selected_groups: Sequence[PrecisionGroup],
            profiles: Mapping[str, str],
            metadata: Mapping[str, Any],
        ) -> CalibrationScheme:
            return scheme_from_profiles(
                scheme,
                evaluations,
                selected_groups,
                profiles,
                metadata=metadata,
            )
    else:
        build_output_scheme = scheme_builder

    if shard_tokens > 0:
        return _refine_soft_rate_distortion_sharded(
            backend,
            corpus,
            validation_corpus or corpus,
            base_scheme,
            groups,
            evaluations,
            build_output_scheme,
            output_scheme,
            report,
            work_dir=work_dir,
            window_length=window_length,
            batch_size=batch_size,
            train_tokens=train_tokens,
            validation_tokens=validation_tokens,
            hard_train_tokens=hard_train_tokens,
            seed=seed,
            shard_tokens=shard_tokens,
            epochs=epochs,
            learning_rate=learning_rate,
            dual_learning_rate=dual_learning_rate,
            initial_dual=initial_dual,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            head_row_chunk=head_row_chunk,
            discrete_iterations=discrete_iterations,
            discrete_single_candidates=discrete_single_candidates,
            discrete_pair_candidates=discrete_pair_candidates,
            validation_tolerance=validation_tolerance,
            resume=resume,
            keep_hidden=keep_hidden,
        )

    if steps <= 0 or train_tokens <= 0 or validation_tokens <= 0:
        raise ValueError("soft-search steps and token counts must be positive")
    if learning_rate <= 0 or dual_learning_rate < 0 or initial_dual < 0:
        raise ValueError("soft-search learning rates and initial dual are invalid")
    if validation_tolerance < 0:
        raise ValueError("validation_tolerance must be non-negative")
    scheme_path = Path(output_scheme).resolve()
    report_path = Path(report).resolve()
    if scheme_path.exists() or report_path.exists():
        raise FileExistsError("soft rate-distortion output already exists")
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    session_work = work / f"soft-session-{os.getpid()}"
    if session_work.exists():
        raise FileExistsError(f"soft-search session directory already exists: {session_work}")
    checkpoint_path = work / "soft-rate-distortion-checkpoint.pt"
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(f"soft-search checkpoint already exists: {checkpoint_path}")
    if resume and not checkpoint_path.is_file():
        raise FileNotFoundError(f"soft-search checkpoint does not exist: {checkpoint_path}")

    if {group.layer for group in groups} != set(range(backend.num_layers)):
        raise ValueError("soft precision groups do not cover every decoder layer")
    fixed_bits = fixed_storage_bits(base_scheme, groups)
    logits = initialize_gate_logits(groups, device=backend.device)
    optimizer = torch.optim.Adam(list(logits.values()), lr=learning_rate)
    start_step = 0
    dual = float(initial_dual)
    steps_report: list[SoftSearchStep] = []
    if resume:
        completed, dual, steps_report = _load_checkpoint(
            checkpoint_path,
            groups,
            logits,
            optimizer,
        )
        start_step = completed + 1
    if start_step >= steps:
        raise ValueError("soft-search checkpoint already completed the requested steps")

    train_batches = list(
        corpus.iter_batches(
            "train",
            window_length=window_length,
            batch_size=batch_size,
            max_tokens=train_tokens,
            seed=seed,
        )
    )
    validation_batches = list(
        corpus.iter_batches(
            "validation",
            window_length=window_length,
            batch_size=batch_size,
            max_tokens=validation_tokens,
            seed=seed + 1,
        )
    )
    minimum_rows = int(getattr(backend, "soft_min_autograd_rows", 0))
    if minimum_rows and any(batch.input_ids.size < minimum_rows for batch in train_batches):
        raise ValueError(
            f"soft-search train batches must contain at least {minimum_rows} rows "
            "so the differentiable dequant+GEMM path is used"
        )
    train_token_count = sum(int(batch.input_ids.size) for batch in train_batches)
    validation_token_count = sum(int(batch.input_ids.size) for batch in validation_batches)
    teacher_bytes = torch.empty((), dtype=backend.teacher_dtype).element_size()
    candidate_bytes = torch.empty((), dtype=backend.quantized_dtype).element_size()
    soft_dtype = getattr(backend, "soft_dtype", backend.quantized_dtype)
    soft_bytes = torch.empty((), dtype=soft_dtype).element_size()
    hidden_disk_bytes = train_token_count * backend.hidden_size * (
        3 * candidate_bytes + backend.num_layers * soft_bytes + 2 * teacher_bytes + 8
    ) + validation_token_count * backend.hidden_size * (3 * candidate_bytes + 2 * teacher_bytes)
    packed_candidate_bytes = (
        sum(option.storage_bits for group in groups for option in group.options) // 8
    )
    print(
        json.dumps(
            {
                "event": "soft_search_contract",
                "model": str(getattr(backend, "root", "custom-backend")),
                "corpus": str(corpus.root),
                "window_length": int(window_length),
                "batch_size": int(batch_size),
                "train_tokens": train_token_count,
                "validation_tokens": validation_token_count,
                "steps": int(steps),
                "objective": "mean_KL(BF16_teacher||relaxed_MFQ)",
                "soft_compute_dtype": str(soft_dtype).removeprefix("torch."),
                "hard_compute_dtype": str(backend.quantized_dtype).removeprefix("torch."),
                "group_count": len(groups),
                "candidate_count": sum(len(group.options) for group in groups),
                "target_storage_bits": int(base_scheme.target_storage_bits),
                "estimated_hidden_disk_bytes": int(hidden_disk_bytes),
                "estimated_packed_candidate_bytes": int(packed_candidate_bytes),
                "output_scheme": str(scheme_path),
                "report": str(report_path),
                "checkpoint": str(checkpoint_path),
            }
        ),
        flush=True,
    )
    hidden_paths: list[Path] = []
    stores: list[HiddenStateStore] = []
    discrete_steps: tuple[DiscreteSearchStep, ...] = ()
    result: SoftRefinementResult | None = None
    try:
        session_work.mkdir()
        train = _create_split_workspace(
            backend,
            session_work,
            "train",
            train_batches,
            hidden_paths,
        )
        stores.extend(
            [
                train.candidate_initial,
                *train.teacher_buffers,
                *train.hard_buffers,
            ]
        )
        validation = _create_split_workspace(
            backend,
            session_work,
            "validation",
            validation_batches,
            hidden_paths,
        )
        stores.extend(
            [
                validation.candidate_initial,
                *validation.teacher_buffers,
                *validation.hard_buffers,
            ]
        )
        _seed_and_run_teacher(backend, (train, validation))

        train_hidden = [train.candidate_initial]
        for layer_index in range(1, backend.num_layers + 1):
            store = _new_store(
                hidden_paths,
                session_work / f"soft-train-hidden-{layer_index:03d}.bin",
                train.token_count,
                backend.hidden_size,
                soft_dtype,
            )
            train_hidden.append(store)
            stores.append(store)
        gradient_store_list: list[HiddenStateStore] = []
        for name in ("a", "b"):
            store = _new_store(
                hidden_paths,
                session_work / f"soft-train-gradient-{name}.bin",
                train.token_count,
                backend.hidden_size,
                torch.float32,
            )
            gradient_store_list.append(store)
            stores.append(store)
        gradient_stores = tuple(gradient_store_list)

        with backend.terminal_objective() as objective:
            for step in range(start_step, steps):
                temperature = _temperature(
                    step,
                    steps,
                    temperature_start,
                    temperature_end,
                )
                optimizer.zero_grad(set_to_none=True)
                _soft_forward(
                    backend,
                    train,
                    train_hidden,
                    groups,
                    logits,
                    temperature,
                )
                mean_kl = _terminal_metric(
                    backend,
                    objective,
                    train,
                    train_hidden[-1],
                    row_chunk=head_row_chunk,
                    gradient_store=gradient_stores[0],
                )
                _soft_backward(
                    backend,
                    train,
                    train_hidden,
                    gradient_stores,  # type: ignore[arg-type]
                    groups,
                    logits,
                    temperature,
                )
                expected_bits = float(
                    expected_storage_bits(
                        groups,
                        logits,
                        temperature=temperature,
                        fixed_bits=fixed_bits,
                    )
                    .detach()
                    .item()
                )
                add_rate_gradient_(
                    groups,
                    logits,
                    temperature=temperature,
                    target_storage_bits=base_scheme.target_storage_bits,
                    dual=dual,
                )
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(list(logits.values()), max_norm=5.0).item()
                )
                optimizer.step()
                dual = update_dual(
                    dual,
                    expected_bits,
                    base_scheme.target_storage_bits,
                    learning_rate=dual_learning_rate,
                )
                item = SoftSearchStep(
                    step=step,
                    temperature=temperature,
                    mean_kl=mean_kl,
                    expected_storage_bits=expected_bits,
                    expected_bpw=expected_bits / base_scheme.weight_count,
                    dual=dual,
                    rate_violation_percent=(
                        100.0
                        * (expected_bits - base_scheme.target_storage_bits)
                        / base_scheme.target_storage_bits
                    ),
                    gradient_norm=gradient_norm,
                )
                steps_report.append(item)
                print(json.dumps({"event": "soft_search_step", **asdict(item)}), flush=True)
                _save_checkpoint(
                    checkpoint_path,
                    step=step,
                    dual=dual,
                    logits=logits,
                    optimizer=optimizer,
                    groups=groups,
                    history=steps_report,
                )

            initial_profiles = discretize_gate_logits(
                groups,
                logits,
                target_storage_bits=base_scheme.target_storage_bits,
                fixed_bits=fixed_bits,
            )
            base_profiles = {group.name: group.base_profile for group in groups}
            train_metric_cache: dict[tuple[tuple[str, str], ...], float] = {}

            def train_metric(profiles: Mapping[str, str]) -> float:
                key = tuple(sorted(profiles.items()))
                if key not in train_metric_cache:
                    train_metric_cache[key] = _hard_metric(
                        backend,
                        objective,
                        train,
                        groups,
                        profiles,
                        row_chunk=head_row_chunk,
                    )
                return train_metric_cache[key]

            refined_profiles, discrete_steps = refine_discrete_profiles(
                groups,
                initial_profiles,
                logits,
                train_metric,
                target_storage_bits=base_scheme.target_storage_bits,
                fixed_bits=fixed_bits,
                max_iterations=discrete_iterations,
                max_single=discrete_single_candidates,
                max_pair=discrete_pair_candidates,
            )
            base_train_kl = train_metric(base_profiles)
            refined_train_kl = train_metric(refined_profiles)
            base_validation_kl = _hard_metric(
                backend,
                objective,
                validation,
                groups,
                base_profiles,
                row_chunk=head_row_chunk,
            )
            refined_validation_kl = _hard_metric(
                backend,
                objective,
                validation,
                groups,
                refined_profiles,
                row_chunk=head_row_chunk,
            )
            train_tolerance = max(1e-12, base_train_kl * validation_tolerance)
            validation_limit = max(1e-12, base_validation_kl * validation_tolerance)
            accepted = (
                refined_train_kl <= base_train_kl + train_tolerance
                and refined_validation_kl <= base_validation_kl + validation_limit
            )
            selected_profiles = refined_profiles if accepted else base_profiles
            selected_train_kl = refined_train_kl if accepted else base_train_kl
            selected_validation_kl = refined_validation_kl if accepted else base_validation_kl

            scheme = build_output_scheme(
                base_scheme,
                groups,
                selected_profiles,
                {
                    "soft_rate_distortion": {
                        "objective": "mean_KL(BF16_teacher||relaxed_MFQ)",
                        "soft_compute_dtype": str(soft_dtype).removeprefix("torch."),
                        "hard_compute_dtype": str(backend.quantized_dtype).removeprefix("torch."),
                        "decision_unit": "per_layer_runtime_precision_group",
                        "train_split": "train",
                        "validation_split": "validation",
                        "window_length": int(window_length),
                        "batch_size": int(batch_size),
                        "train_tokens": int(train.token_count),
                        "train_scored_positions": int(train.scored_positions),
                        "validation_tokens": int(validation.token_count),
                        "validation_scored_positions": int(validation.scored_positions),
                        "steps": int(steps),
                        "learning_rate": float(learning_rate),
                        "dual_learning_rate": float(dual_learning_rate),
                        "temperature_start": float(temperature_start),
                        "temperature_end": float(temperature_end),
                        "head_row_chunk": int(head_row_chunk),
                        "group_count": len(groups),
                        "accepted_on_validation": accepted,
                        "base_train_kl": base_train_kl,
                        "selected_train_kl": selected_train_kl,
                        "base_validation_kl": base_validation_kl,
                        "selected_validation_kl": selected_validation_kl,
                    }
                },
            )
            save_scheme(scheme_path, scheme)
            loaded = load_scheme(scheme_path)

            report_document = {
                "format": "mfq.soft-rate-distortion.v1",
                "base_scheme": str(base_scheme.path),
                "output_scheme": str(scheme_path),
                "corpus": str(corpus.root),
                "target_storage_bits": int(base_scheme.target_storage_bits),
                "actual_storage_bits": int(loaded.storage_bits),
                "actual_bpw": loaded.bpw,
                "fixed_storage_bits": int(fixed_bits),
                "group_count": len(groups),
                "candidate_count": sum(len(group.options) for group in groups),
                "soft_compute_dtype": str(soft_dtype).removeprefix("torch."),
                "hard_compute_dtype": str(backend.quantized_dtype).removeprefix("torch."),
                "accepted_on_validation": accepted,
                "base_train_kl": base_train_kl,
                "refined_train_kl": refined_train_kl,
                "selected_train_kl": selected_train_kl,
                "base_validation_kl": base_validation_kl,
                "refined_validation_kl": refined_validation_kl,
                "selected_validation_kl": selected_validation_kl,
                "initial_train_kl": discrete_steps[0].metric,
                "steps": [asdict(item) for item in steps_report],
                "discrete_steps": [asdict(item) for item in discrete_steps],
                "initial_profiles": dict(sorted(initial_profiles.items())),
                "refined_profiles": dict(sorted(refined_profiles.items())),
                "selected_profiles": dict(sorted(selected_profiles.items())),
                "learned_logits": {
                    group.name: {
                        option.profile: float(logits[group.name][index].detach().cpu().item())
                        for index, option in enumerate(group.options)
                    }
                    for group in groups
                },
                "selected_storage_bits": selected_storage_bits(
                    groups,
                    selected_profiles,
                    fixed_bits=fixed_bits,
                ),
                "initial_storage_bits": selected_storage_bits(
                    groups,
                    initial_profiles,
                    fixed_bits=fixed_bits,
                ),
                "refined_storage_bits": selected_storage_bits(
                    groups,
                    refined_profiles,
                    fixed_bits=fixed_bits,
                ),
            }
            report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = report_path.with_suffix(report_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(report_document, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, report_path)
            result = SoftRefinementResult(
                scheme=loaded,
                steps=tuple(steps_report),
                discrete_steps=discrete_steps,
                base_train_kl=base_train_kl,
                selected_train_kl=selected_train_kl,
                base_validation_kl=base_validation_kl,
                selected_validation_kl=selected_validation_kl,
                accepted=accepted,
            )
    finally:
        backend.release_initial_state()
        backend.clear_quantized_cache()
        for store in reversed(stores):
            store.close()
        if not keep_hidden:
            for path in hidden_paths:
                path.unlink(missing_ok=True)
            if session_work.exists():
                session_work.rmdir()
    if result is None:
        raise RuntimeError("soft rate-distortion refinement produced no result")
    return result


__all__ = [
    "SoftRefinementBackend",
    "SoftRefinementResult",
    "SoftSearchStep",
    "refine_soft_rate_distortion",
]
