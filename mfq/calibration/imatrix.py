"""Architecture-aware BF16 activation-imatrix collection on CUDA and Metal."""

from __future__ import annotations

import gc
import hashlib
import json
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from mfq.calibration.collector import HiddenStateStore
from mfq.calibration.dataset import CalibrationBatch, CalibrationCorpus
from mfq.quantize.imatrix import ImportanceEntry, ImportanceMatrix, save_importance_matrix


@dataclass(frozen=True)
class ImatrixTarget:
    name: str
    module_name: str
    width: int
    experts: int = 1
    kind: str = "linear"

    def __post_init__(self) -> None:
        if not self.name or not self.module_name or self.width <= 0 or self.experts <= 0:
            raise ValueError(f"invalid imatrix target: {self}")
        if self.kind not in {"linear", "expert_gate_up", "expert_down"}:
            raise ValueError(f"unsupported imatrix target kind: {self.kind}")


class ActivationImatrixCollector:
    """Accumulate E[x^2] with independent counters for routed experts."""

    def __init__(
        self,
        targets: Sequence[ImatrixTarget],
        device: torch.device,
        *,
        accumulation_dtype: torch.dtype = torch.float64,
    ) -> None:
        if accumulation_dtype not in {torch.float32, torch.float64}:
            raise ValueError("imatrix accumulation dtype must be float32 or float64")
        self.targets = tuple(targets)
        self.device = device
        self.accumulation_dtype = accumulation_dtype
        self.sums = {
            target.name: torch.zeros(
                (target.experts, target.width),
                device=device,
                dtype=accumulation_dtype,
            )
            for target in targets
        }
        self.counts = {
            target.name: torch.zeros(target.experts, device=device, dtype=torch.int64)
            for target in targets
        }
        self.handles: list[Any] = []
        self.restores: list[tuple[nn.Module, Any]] = []
        self.valid_mask: torch.Tensor | None = None

    def set_valid_mask(self, value: torch.Tensor | None) -> None:
        self.valid_mask = None if value is None else value.detach().reshape(-1).to(torch.bool)

    def _matrix(self, value: torch.Tensor, width: int, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.shape[-1] != width:
            shape = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
            raise ValueError(f"imatrix input width mismatch for {name}: {shape} vs {width}")
        matrix = value.detach().reshape(-1, width).float()
        if self.valid_mask is not None and self.valid_mask.numel() == matrix.shape[0]:
            matrix = matrix[self.valid_mask]
        return matrix

    def add_linear(self, target: ImatrixTarget, value: torch.Tensor) -> None:
        matrix = self._matrix(value, target.width, target.name)
        self.sums[target.name][0].add_(matrix.square().sum(0, dtype=self.accumulation_dtype))
        self.counts[target.name][0].add_(int(matrix.shape[0]))

    def add_experts(
        self,
        target: ImatrixTarget,
        value: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> None:
        matrix = self._matrix(value, target.width, target.name)
        selected = selected_experts.detach().reshape(matrix.shape[0], -1).to(torch.int64)
        if selected.numel() and (
            int(selected.min().item()) < 0 or int(selected.max().item()) >= target.experts
        ):
            raise IndexError(f"routed expert index is outside {target.experts} for {target.name}")
        token_ids = (
            torch.arange(matrix.shape[0], device=matrix.device, dtype=torch.int64)
            .unsqueeze(1)
            .expand_as(selected)
            .reshape(-1)
        )
        expert_ids = selected.reshape(-1)
        routed = matrix.index_select(0, token_ids).square().to(self.accumulation_dtype)
        self.sums[target.name].index_add_(0, expert_ids, routed)
        self.counts[target.name].add_(
            torch.bincount(expert_ids, minlength=target.experts)
        )

    def add_expert(
        self,
        target: ImatrixTarget,
        value: torch.Tensor,
        expert: int,
    ) -> None:
        matrix = self._matrix(value, target.width, target.name)
        self.sums[target.name][expert].add_(
            matrix.square().sum(0, dtype=self.accumulation_dtype)
        )
        self.counts[target.name][expert].add_(int(matrix.shape[0]))

    def install_layer(
        self,
        layer: nn.Module,
        layer_index: int,
        targets: Sequence[ImatrixTarget],
    ) -> None:
        modules = dict(layer.named_modules())
        by_module: dict[str, list[ImatrixTarget]] = {}
        for target in targets:
            by_module.setdefault(target.module_name, []).append(target)
        for module_name, module_targets in by_module.items():
            try:
                module = modules[module_name]
            except KeyError as exc:
                raise ValueError(
                    f"layer {layer_index} lacks imatrix module {module_name!r}"
                ) from exc
            kinds = {target.kind for target in module_targets}
            if kinds == {"linear"}:
                if len(module_targets) != 1:
                    raise TypeError(f"imatrix target is not one matrix projection: {module_name}")
                target = module_targets[0]

                def pre_hook(_module, inputs, *, _target=target):
                    self.add_linear(_target, inputs[0])

                self.handles.append(module.register_forward_pre_hook(pre_hook))
                continue
            if kinds != {"expert_gate_up", "expert_down"} or len(module_targets) != 2:
                raise TypeError(f"invalid routed-expert imatrix binding: {module_name}")
            gate = next(item for item in module_targets if item.kind == "expert_gate_up")
            down = next(item for item in module_targets if item.kind == "expert_down")
            gate_up = getattr(module, "gate_up_proj", None)
            down_proj = getattr(module, "down_proj", None)
            activation = getattr(module, "act_fn", None)
            if (
                not isinstance(gate_up, torch.Tensor)
                or not isinstance(down_proj, torch.Tensor)
                or not callable(activation)
            ):
                raise TypeError(f"unsupported routed-expert module: {module_name}")
            original = module.forward

            def expert_forward(
                _module,
                hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                top_k_weights: torch.Tensor,
                *,
                _gate=gate,
                _down=down,
            ) -> torch.Tensor:
                # Mirror the Transformers eager expert implementation, while
                # collecting the actual Gate/Up and Down inputs in the same
                # pass. This avoids a second Gate/Up matmul and a potentially
                # enormous [tokens, top_k, 2I, H] gathered-weight temporary.
                final = torch.zeros_like(hidden_states)
                with torch.no_grad():
                    # One accelerator-to-host transfer avoids one synchronous
                    # ``item()`` call per active expert.  Gemma4 commonly hits
                    # all 128 experts in every calibration batch.
                    active = torch.unique(selected_experts).to(torch.int64).cpu().tolist()
                for expert in active:
                    expert = int(expert)
                    token_idx, top_k_pos = torch.where(selected_experts == expert)
                    current = hidden_states[token_idx]
                    valid = None
                    if self.valid_mask is not None:
                        valid = self.valid_mask.index_select(0, token_idx)
                    measured_current = current if valid is None else current[valid]
                    if measured_current.numel():
                        self.add_expert(_gate, measured_current, expert)
                    gate_value, up_value = torch.nn.functional.linear(
                        current, _module.gate_up_proj[expert]
                    ).chunk(2, dim=-1)
                    intermediate = _module.act_fn(gate_value) * up_value
                    measured_intermediate = intermediate if valid is None else intermediate[valid]
                    if measured_intermediate.numel():
                        self.add_expert(_down, measured_intermediate, expert)
                    output = torch.nn.functional.linear(
                        intermediate, _module.down_proj[expert]
                    )
                    output = output * top_k_weights[token_idx, top_k_pos, None]
                    final.index_add_(0, token_idx, output.to(final.dtype))
                return final

            self.restores.append((module, original))
            module.forward = types.MethodType(expert_forward, module)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for module, original in reversed(self.restores):
            module.forward = original
        self.restores.clear()

    def entries(self) -> dict[str, ImportanceEntry]:
        result: dict[str, ImportanceEntry] = {}
        for target in self.targets:
            counts = self.counts[target.name].detach().cpu().numpy().astype(np.int64)
            sums = self.sums[target.name].detach().cpu().numpy().astype(np.float64)
            values = np.ones_like(sums, dtype=np.float32)
            positive = counts > 0
            values[positive] = (sums[positive] / counts[positive, None]).astype(np.float32)
            if not positive.any():
                raise RuntimeError(f"imatrix target received no activations: {target.name}")
            result[target.name] = ImportanceEntry(np.ascontiguousarray(values), counts)
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
    return {"name": root.name, "files": files}


def _release(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()


def _backend(model: Path, device: torch.device, attention: str):
    from transformers import AutoConfig

    outer = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    config = getattr(outer, "text_config", outer)
    model_type = str(getattr(config, "model_type", ""))
    if model_type == "gemma4_text":
        from mfq.calibration.layerwise_gemma4 import Gemma4LayerwiseBackend

        return Gemma4LayerwiseBackend(model, device=device, attention=attention), model_type
    if model_type in {"qwen3_5", "qwen3_5_moe"}:
        from mfq.calibration.layerwise_qwen35 import Qwen35LayerwiseBackend

        return (
            Qwen35LayerwiseBackend(
                model, None, device=device, quant_backend="cpu", attention=attention
            ),
            model_type,
        )
    raise ValueError(f"generic imatrix does not yet support model type {model_type!r}")


def _targets(backend: Any, model_type: str) -> tuple[dict[int, tuple[ImatrixTarget, ...]], tuple[ImatrixTarget, ...]]:
    by_layer: dict[int, list[ImatrixTarget]] = {}
    index = backend.index
    for layer in range(backend.num_layers):
        prefix = f"model.language_model.layers.{layer}."
        values: list[ImatrixTarget] = []
        for name in sorted(index.weight_map):
            if not name.startswith(prefix):
                continue
            shape = index.shape(name)
            suffix = name[len(prefix) :]
            if len(shape) == 2 and suffix.endswith(".weight"):
                module_name = suffix.removesuffix(".weight")
                values.append(ImatrixTarget(name, module_name, int(shape[1])))
        if model_type in {"gemma4_text", "qwen3_5_moe"}:
            expert_module = "experts" if model_type == "gemma4_text" else "mlp.experts"
            gate_name = prefix + expert_module + ".gate_up_proj"
            down_name = prefix + expert_module + ".down_proj"
            if gate_name in index.weight_map and down_name in index.weight_map:
                gate_shape = index.shape(gate_name)
                down_shape = index.shape(down_name)
                values.extend(
                    (
                        ImatrixTarget(
                            gate_name,
                            expert_module,
                            int(gate_shape[2]),
                            int(gate_shape[0]),
                            "expert_gate_up",
                        ),
                        ImatrixTarget(
                            down_name,
                            expert_module,
                            int(down_shape[2]),
                            int(down_shape[0]),
                            "expert_down",
                        ),
                    )
                )
        by_layer[layer] = values
    all_targets = tuple(target for layer in range(backend.num_layers) for target in by_layer[layer])
    if not all_targets:
        raise ValueError("model contains no supported imatrix targets")
    return {key: tuple(value) for key, value in by_layer.items()}, all_targets


def collect_imatrix(
    model_path: str | Path,
    corpus: CalibrationCorpus,
    output: str | Path,
    *,
    device: str = "cuda:0",
    attention: str = "sdpa",
    window_length: int = 16_384,
    batch_size: int = 1,
    pad_to_multiple: int | None = None,
    train_tokens: int = 1_572_864,
    seed: int = 20260810,
    work_dir: str | Path | None = None,
    keep_hidden: bool = False,
    accumulation_dtype: str = "float64",
) -> ImportanceMatrix:
    """Collect one frozen train-only BF16 activation imatrix layer by layer."""

    root = Path(model_path).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"imatrix already exists: {output_path}")
    target_device = torch.device(device)
    if target_device.type not in {"cuda", "mps"}:
        raise ValueError("imatrix device must be CUDA or MPS")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA imatrix requested but CUDA is unavailable")
    if target_device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Metal imatrix requested but MPS is unavailable")
    if min(window_length, batch_size, train_tokens) <= 0:
        raise ValueError("imatrix window, batch, and train token counts must be positive")
    dtype = {"float32": torch.float32, "float64": torch.float64}.get(accumulation_dtype)
    if dtype is None:
        raise ValueError("accumulation dtype must be float32 or float64")
    # MPS does not expose float64 arithmetic. FP32 accumulation remains much
    # more precise than the BF16 forward values and is the Metal default.
    if target_device.type == "mps" and dtype == torch.float64:
        dtype = torch.float32

    batches = tuple(
        corpus.iter_batches(
            "train",
            window_length=window_length,
            batch_size=batch_size,
            max_tokens=train_tokens,
            seed=seed,
            drop_last=False,
            pad_to_multiple=pad_to_multiple,
        )
    )
    if not batches:
        raise ValueError("training corpus produced no imatrix batches")
    selected_tokens = sum(int(batch.attention_mask.sum()) for batch in batches)
    storage_tokens = sum(int(batch.input_ids.size) for batch in batches)
    work = (
        Path(work_dir).resolve()
        if work_dir is not None
        else output_path.parent / f"{output_path.stem}.work"
    )
    work.mkdir(parents=True, exist_ok=True)
    hidden_path = work / "hidden.bf16"
    if hidden_path.exists():
        raise FileExistsError(f"imatrix hidden state already exists: {hidden_path}")
    backend, model_type = _backend(root, target_device, attention)
    targets_by_layer, targets = _targets(backend, model_type)
    collector = ActivationImatrixCollector(targets, target_device, accumulation_dtype=dtype)
    store = HiddenStateStore(
        hidden_path,
        storage_tokens,
        backend.hidden_size,
        backend.teacher_dtype,
    )
    layout: list[tuple[CalibrationBatch, int, int]] = []
    cursor = 0
    started = time.time()
    try:
        for batch in batches:
            ids = torch.as_tensor(batch.input_ids, dtype=torch.int64)
            value = backend.initial_hidden(ids)
            end = cursor + int(ids.numel())
            store.write(cursor, value)
            layout.append((batch, cursor, end))
            cursor = end
        store.flush()
        backend.release_initial_state()
        for layer_index in range(backend.num_layers):
            with backend.layer(layer_index, quantized=False) as layer:
                collector.install_layer(layer, layer_index, targets_by_layer[layer_index])
                try:
                    for batch, start, end in layout:
                        hidden = store.read(
                            start,
                            end,
                            batch.input_ids.shape,
                            device=target_device,
                        )
                        valid_mask = torch.as_tensor(
                            batch.attention_mask,
                            device=target_device,
                            dtype=torch.bool,
                        )
                        collector.set_valid_mask(valid_mask)
                        store.write(
                            start,
                            backend.forward_layer(
                                layer,
                                layer_index,
                                hidden,
                                attention_mask=valid_mask,
                            ),
                        )
                finally:
                    collector.set_valid_mask(None)
                    collector.close()
            store.flush()
            _release(target_device)
            print(
                json.dumps(
                    {
                        "event": "imatrix_layer",
                        "layer": layer_index,
                        "layers": backend.num_layers,
                        "tokens": selected_tokens,
                        "device": str(target_device),
                        "seconds": round(time.time() - started, 3),
                    }
                ),
                flush=True,
            )
        entries = collector.entries()
        # Qwen3.5 stores linear-attention Q/K/V in one source matrix but the
        # quantization plan exposes Q/K and V as two transformed tensors. Both
        # consume the same input activation, so preserve explicit aliases.
        for name, entry in tuple(entries.items()):
            if name.endswith(".linear_attn.in_proj_qkv.weight"):
                base = name[: -len("in_proj_qkv.weight")]
                entries[base + "in_proj_qk.weight"] = entry
                entries[base + "in_proj_v.weight"] = entry
        metadata = {
            "objective": "mean_squared_linear_input_activation",
            "split": "train",
            "model": _model_identity(root),
            "model_type": model_type,
            "corpus": {
                "name": corpus.root.name,
                "manifest_sha256": _sha256(corpus.root / "manifest.json"),
            },
            "device": str(target_device),
            "backend": "metal" if target_device.type == "mps" else "cuda",
            "forward_dtype": "bfloat16",
            "accumulation_dtype": str(dtype).removeprefix("torch."),
            "attention": attention,
            "seed": int(seed),
            "window_length": int(window_length),
            "batch_size": int(batch_size),
            "pad_to_multiple": None if pad_to_multiple is None else int(pad_to_multiple),
            "storage_tokens": int(storage_tokens),
            "tokens": int(selected_tokens),
            "targets": len(entries),
            "routed_expert_entries": sum(entry.matrices > 1 for entry in entries.values()),
            "elapsed_seconds": time.time() - started,
        }
        result = save_importance_matrix(
            output_path,
            entries,
            datasets=(corpus.root.name,),
            chunk_count=len(batches),
            chunk_size=window_length,
            metadata=metadata,
        )
        print(
            json.dumps(
                {
                    "event": "imatrix_saved",
                    "output": str(output_path),
                    "entries": len(entries),
                    "tokens": selected_tokens,
                    "bytes": output_path.stat().st_size,
                    "seconds": round(time.time() - started, 3),
                }
            ),
            flush=True,
        )
        return result
    finally:
        collector.close()
        backend.release_initial_state()
        close = getattr(backend, "close", None)
        if callable(close):
            close()
        store.close()
        if not keep_hidden:
            hidden_path.unlink(missing_ok=True)
        _release(target_device)


__all__ = [
    "ActivationImatrixCollector",
    "ImatrixTarget",
    "collect_imatrix",
]
