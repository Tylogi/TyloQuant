"""Qwen3.5 backend for disk-backed layerwise calibration replay."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch import nn

from mfq.calibration.artifact import CalibrationScheme
from mfq.calibration.qwen35 import (
    HfSafetensorIndex,
    Qwen35LinearTarget,
    qwen35_linear_targets,
)
from mfq.calibration.rate_distortion import PrecisionGroup
from mfq.calibration.terminal_kl import ChunkedTerminalObjective
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import NintTensor
from mfq.quantize.nint_quant import quantize as quantize_nint_cpu
from mfq.quantize.nint_quant_torch import quantize_axis0 as quantize_nint_cuda
from mfq.runtime.torch_linear import TorchNintLinear


class _NintLinearModule(nn.Module):
    def __init__(
        self,
        linear: TorchNintLinear,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.linear = linear.shared_weights_clone()
        self.register_buffer(
            "bias",
            None if bias is None else bias.to(device=self.linear.device, dtype=torch.float16),
            persistent=False,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.linear(value)
        if self.bias is not None:
            output = output + self.bias
        return output


class _SplitNintLinearModule(nn.Module):
    def __init__(self, modules: Sequence[_NintLinearModule]) -> None:
        super().__init__()
        if len(modules) < 2:
            raise ValueError("split NINT linear requires at least two row segments")
        self.parts = nn.ModuleList(modules)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.cat([part(value) for part in self.parts], dim=-1)


class _SoftNintLinearModule(nn.Module):
    def __init__(
        self,
        modules: Sequence[_NintLinearModule],
        logits: torch.Tensor,
        temperature: float,
    ) -> None:
        super().__init__()
        if len(modules) < 2 or logits.shape != (len(modules),):
            raise ValueError("soft NINT linear requires one logit per candidate")
        if temperature <= 0:
            raise ValueError("soft NINT temperature must be positive")
        biases = [module.bias for module in modules]
        if any((bias is None) != (biases[0] is None) for bias in biases[1:]):
            raise ValueError("soft NINT candidates disagree on bias presence")
        self.register_buffer("bias", biases[0], persistent=False)
        self.weight_names: list[str] = []
        for index, module in enumerate(modules):
            name = f"candidate_weight_{index}"
            self.register_buffer(name, module.linear.weight.detach(), persistent=False)
            self.weight_names.append(name)
        self.logits = logits
        self.temperature = float(temperature)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(self.logits / self.temperature, dim=0)
        accumulated: torch.Tensor | None = None
        output_dtype: torch.dtype | None = None
        for index, name in enumerate(self.weight_names):
            weight = getattr(self, name)
            output = torch.nn.functional.linear(
                value.to(dtype=weight.dtype),
                weight,
                self.bias,
            )
            output_dtype = output.dtype
            term = output.float() * probabilities[index]
            accumulated = term if accumulated is None else accumulated + term
        if accumulated is None or output_dtype is None:
            raise RuntimeError("soft NINT linear produced no candidate outputs")
        return accumulated.to(dtype=output_dtype)


class _SplitSoftNintLinearModule(nn.Module):
    def __init__(self, modules: Sequence[nn.Module]) -> None:
        super().__init__()
        if len(modules) < 2:
            raise ValueError("split soft NINT linear requires at least two row segments")
        self.parts = nn.ModuleList(modules)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.cat([part(value) for part in self.parts], dim=-1)


class _OpenSafetensorRows:
    def __init__(self, reader, name: str) -> None:
        self.reader = reader
        self.name = name
        self.shape = tuple(int(value) for value in reader.get_slice(name).get_shape())

    def rows(self, start: int, end: int) -> torch.Tensor:
        return self.reader.get_slice(self.name)[start:end].contiguous()


def _module(root: nn.Module, name: str) -> nn.Module:
    current = root
    for part in name.split("."):
        current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise TypeError(f"Qwen3.5 calibration target is not a module: {name}")
    return current


def _replace_module(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = _module(root, parent_name) if parent_name else root
    setattr(parent, child, replacement)


class Qwen35LayerwiseBackend:
    """Load one original or NINT Qwen3.5 decoder layer at a time."""

    teacher_dtype = torch.bfloat16
    quantized_dtype = torch.float16
    soft_dtype = torch.bfloat16
    # Soft candidates are materialized as dense FP16 weights before linear(),
    # so autograd remains available for short and partial calibration batches.
    soft_min_autograd_rows = 1

    def __init__(
        self,
        model_path: str | Path,
        scheme: CalibrationScheme | None,
        *,
        device: str | torch.device = "cuda:0",
        quant_backend: str = "cuda",
        attention: str = "sdpa",
        candidate_cache_dir: str | Path | None = None,
    ) -> None:
        try:
            from transformers import AutoConfig
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Qwen3.5 layerwise replay requires transformers") from exc

        self.root = Path(model_path).resolve()
        self.scheme = scheme
        self.device = torch.device(device)
        self.quant_backend = quant_backend
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Qwen3.5 layerwise replay requested CUDA without CUDA support")
        if quant_backend not in {"cuda", "cpu"}:
            raise ValueError("quant_backend must be cuda or cpu")
        if quant_backend == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA quantization requested without CUDA support")
        if attention not in {"eager", "sdpa"}:
            raise ValueError("attention must be eager or sdpa")

        outer_config = AutoConfig.from_pretrained(self.root, trust_remote_code=True)
        self.config = getattr(outer_config, "text_config", outer_config)
        self.is_moe = int(getattr(self.config, "num_experts", 0)) > 0
        self.config._attn_implementation = attention
        self.num_layers = int(self.config.num_hidden_layers)
        self.hidden_size = int(self.config.hidden_size)
        self.index = HfSafetensorIndex(self.root)
        if self.is_moe:
            if self.scheme is not None:
                raise NotImplementedError(
                    "Qwen3.5-MoE quantized layerwise calibration uses the expert-wise backend"
                )
            targets: Sequence[Qwen35LinearTarget] = ()
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeTextRotaryEmbedding,
            )

            rotary_type = Qwen3_5MoeTextRotaryEmbedding
        else:
            targets = qwen35_linear_targets(self.root)
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5TextRotaryEmbedding,
            )

            rotary_type = Qwen3_5TextRotaryEmbedding
        self.targets_by_layer: dict[int, list[Qwen35LinearTarget]] = defaultdict(list)
        for target in targets:
            layer = int(target.name.split(".layers.", 1)[1].split(".", 1)[0])
            self.targets_by_layer[layer].append(target)
            if self.scheme is not None:
                self.scheme.require(target.name)
        self.rotary = rotary_type(self.config, device=self.device).to(self.device)
        self.candidate_cache_dir = (
            None if candidate_cache_dir is None else Path(candidate_cache_dir).resolve()
        )
        if self.candidate_cache_dir is not None:
            self.candidate_cache_dir.mkdir(parents=True, exist_ok=True)
        self._embedding: torch.Tensor | None = None
        self._gpu_encoded_cache: dict[tuple[str, NintSpec], TorchNintLinear] = {}

    def initial_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._embedding is None:
            self._embedding = self.index.tensor(
                "model.language_model.embed_tokens.weight",
                device="cpu",
            )
        ids = input_ids.detach().to(device="cpu", dtype=torch.int64)
        hidden = self._embedding.index_select(0, ids.reshape(-1)).reshape(
            *ids.shape, self.hidden_size
        )
        return hidden.to(device=self.device, dtype=self.teacher_dtype)

    def release_initial_state(self) -> None:
        self._embedding = None
        gc.collect()

    def _load_dense_layer(self, layer_index: int, dtype: torch.dtype) -> nn.Module:
        if self.is_moe:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeDecoderLayer as DecoderLayer,
            )
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5DecoderLayer as DecoderLayer,
            )

        with torch.device("meta"):
            layer = DecoderLayer(self.config, layer_index)
        state = dict(self.index.layer_state(layer_index))
        result = layer.load_state_dict(state, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Qwen3.5 layer {layer_index} state mismatch: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        return layer.to(device=self.device, dtype=dtype).eval().requires_grad_(False)

    def _quantize(self, weight: torch.Tensor, spec: NintSpec) -> NintTensor:
        if self.quant_backend == "cuda":
            return quantize_nint_cuda(weight, spec, device=self.device)
        return quantize_nint_cpu(weight.float().cpu().numpy(), spec, axis=0)

    def _candidate_cache_path(
        self,
        target: Qwen35LinearTarget,
        spec: NintSpec,
    ) -> Path | None:
        if self.candidate_cache_dir is None:
            return None
        shard = self.root / self.index.weight_map[target.source_name]
        stat = shard.stat()
        identity = "|".join(
            (
                str(self.root),
                target.source_name,
                str(target.row_start),
                str(target.row_end),
                str(spec.bits),
                str(spec.groupsize),
                str(spec.sub_bits),
                str(stat.st_size),
                str(stat.st_mtime_ns),
            )
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self.candidate_cache_dir / f"{digest}.nint-exec.npz"

    def _load_cached_candidate(
        self,
        target: Qwen35LinearTarget,
        spec: NintSpec,
    ) -> TorchNintLinear | None:
        path = self._candidate_cache_path(target, spec)
        if path is None or not path.is_file():
            return None
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
                raise ValueError(f"invalid packed calibration candidate: {path}")
            arrays = {
                name: np.ascontiguousarray(archive[name])
                for name in (
                    "q_packed",
                    "sub_scale",
                    "sub_min",
                    "neuron_scale",
                    "neuron_min",
                )
            }
        return TorchNintLinear.from_deploy_arrays(
            arrays,
            spec=spec,
            shape=(target.rows, target.columns),
            axis=0,
            neuron_len=target.columns,
            device=self.device,
        )

    def _save_cached_candidate(
        self,
        target: Qwen35LinearTarget,
        encoded: NintTensor,
    ) -> None:
        path = self._candidate_cache_path(target, encoded.spec)
        if path is None or path.exists():
            return
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        if temporary.exists():
            raise FileExistsError(f"packed candidate temporary file already exists: {temporary}")
        metadata = np.frombuffer(
            json.dumps(
                {
                    "name": target.name,
                    "bits": encoded.spec.bits,
                    "groupsize": encoded.spec.groupsize,
                    "sub_bits": encoded.spec.sub_bits,
                    "shape": list(encoded.shape),
                    "axis": encoded.axis,
                    "neuron_len": encoded.neuron_len,
                },
                separators=(",", ":"),
            ).encode("utf-8"),
            dtype=np.uint8,
        )
        with temporary.open("xb") as stream:
            np.savez(
                stream,
                metadata=metadata,
                **TorchNintLinear.deploy_arrays(encoded),
            )
        os.replace(temporary, path)

    def _prepare_layer_candidates(
        self,
        layer_index: int,
        required: Mapping[str, set[NintSpec]],
    ) -> None:
        targets = self.targets_by_layer[layer_index]
        expected = {target.name for target in targets}
        if set(required) != expected or any(not specs for specs in required.values()):
            raise ValueError(f"layer {layer_index} candidate specs do not cover every target")

        by_source: dict[str, list[Qwen35LinearTarget]] = defaultdict(list)
        for target in targets:
            by_source[target.source_name].append(target)
        for source_name, source_targets in sorted(by_source.items()):
            missing = [
                (target, spec)
                for target in source_targets
                for spec in sorted(
                    required[target.name],
                    key=lambda value: (value.bits, value.groupsize, value.sub_bits),
                )
                if (target.name, spec) not in self._gpu_encoded_cache
            ]
            if not missing:
                continue
            source: torch.Tensor | None = None
            try:
                for target, spec in missing:
                    linear = self._load_cached_candidate(target, spec)
                    if linear is None:
                        if source is None:
                            source = self.index.tensor(source_name, device="cpu")
                        weight = source[target.row_start : target.row_end]
                        encoded = self._quantize(weight, spec)
                        self._save_cached_candidate(target, encoded)
                        linear = TorchNintLinear(encoded, self.device)
                        del encoded
                    self._gpu_encoded_cache[(target.name, spec)] = linear
            finally:
                del source

    def prepare_layer_strategies(
        self,
        layer_index: int,
        strategies: Sequence[Mapping[str, NintSpec]],
    ) -> None:
        targets = self.targets_by_layer[layer_index]
        expected = {target.name for target in targets}
        required: dict[str, set[NintSpec]] = defaultdict(set)
        for specs in strategies:
            if set(specs) != expected:
                raise ValueError(f"layer {layer_index} candidate does not cover every target")
            for name, spec in specs.items():
                required[name].add(spec)
        self._prepare_layer_candidates(layer_index, required)

    def _load_quantized_layer(
        self,
        layer_index: int,
        specs: Mapping[str, NintSpec] | None = None,
    ) -> nn.Module:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

        if self.scheme is None:
            raise RuntimeError("quantized layer loading requires a calibration scheme")
        targets_for_layer = self.targets_by_layer[layer_index]
        resolved_specs = {
            target.name: (
                self.scheme.require(target.name).spec if specs is None else specs[target.name]
            )
            for target in targets_for_layer
        }
        self.prepare_layer_strategies(layer_index, [resolved_specs])
        with torch.device("meta"):
            layer = Qwen3_5DecoderLayer(self.config, layer_index)

        prefix = f"model.layers.{layer_index}."
        by_module: dict[str, list[Qwen35LinearTarget]] = defaultdict(list)
        for target in targets_for_layer:
            if not target.module_name.startswith(prefix):
                raise ValueError(f"invalid local module binding for {target.name}")
            by_module[target.module_name[len(prefix) :]].append(target)

        excluded: set[str] = {target.source_name for target in targets_for_layer}
        for module_name, targets in sorted(by_module.items()):
            original = _module(layer, module_name)
            if not isinstance(original, nn.Linear):
                raise TypeError(f"Qwen3.5 quantized target is not Linear: {module_name}")
            bias_name = targets[0].source_name.removesuffix(".weight") + ".bias"
            source_bias = (
                self.index.tensor(bias_name, device="cpu")
                if bias_name in self.index.weight_map
                else None
            )
            if source_bias is not None:
                excluded.add(bias_name)
            parts: list[_NintLinearModule] = []
            for target in sorted(targets, key=lambda item: item.row_start):
                spec = resolved_specs[target.name]
                cache_key = (target.name, spec)
                linear = self._gpu_encoded_cache[cache_key]
                bias = (
                    None if source_bias is None else source_bias[target.row_start : target.row_end]
                )
                parts.append(_NintLinearModule(linear, bias))
            replacement: nn.Module = parts[0] if len(parts) == 1 else _SplitNintLinearModule(parts)
            _replace_module(layer, module_name, replacement)

        state = dict(self.index.layer_state(layer_index, exclude=excluded))
        result = layer.load_state_dict(state, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Qwen3.5 quantized layer {layer_index} state mismatch: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )

        return layer.to(device=self.device, dtype=self.quantized_dtype).eval()

    def _load_soft_layer(
        self,
        layer_index: int,
        groups: Sequence[PrecisionGroup],
        logits: Mapping[str, torch.Tensor],
        temperature: float,
    ) -> nn.Module:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

        targets_for_layer = self.targets_by_layer[layer_index]
        layer_groups = [group for group in groups if group.layer == layer_index]
        group_by_tensor = {name: group for group in layer_groups for name in group.tensor_names}
        expected = {target.name for target in targets_for_layer}
        if set(group_by_tensor) != expected:
            missing = sorted(expected - set(group_by_tensor))
            extra = sorted(set(group_by_tensor) - expected)
            raise ValueError(
                f"soft layer {layer_index} precision groups mismatch: "
                f"missing={missing[:4]}, extra={extra[:4]}"
            )
        if any(group.name not in logits for group in layer_groups):
            raise ValueError(f"soft layer {layer_index} has missing gate logits")

        required = {
            target.name: {
                option.specs[target.name] for option in group_by_tensor[target.name].options
            }
            for target in targets_for_layer
        }
        self._prepare_layer_candidates(layer_index, required)
        with torch.device("meta"):
            layer = Qwen3_5DecoderLayer(self.config, layer_index)

        prefix = f"model.layers.{layer_index}."
        by_module: dict[str, list[Qwen35LinearTarget]] = defaultdict(list)
        for target in targets_for_layer:
            if not target.module_name.startswith(prefix):
                raise ValueError(f"invalid local module binding for {target.name}")
            by_module[target.module_name[len(prefix) :]].append(target)

        excluded: set[str] = {target.source_name for target in targets_for_layer}
        for module_name, targets in sorted(by_module.items()):
            original = _module(layer, module_name)
            if not isinstance(original, nn.Linear):
                raise TypeError(f"Qwen3.5 soft target is not Linear: {module_name}")
            bias_name = targets[0].source_name.removesuffix(".weight") + ".bias"
            source_bias = (
                self.index.tensor(bias_name, device="cpu")
                if bias_name in self.index.weight_map
                else None
            )
            if source_bias is not None:
                excluded.add(bias_name)
            parts: list[nn.Module] = []
            for target in sorted(targets, key=lambda item: item.row_start):
                group = group_by_tensor[target.name]
                bias = (
                    None if source_bias is None else source_bias[target.row_start : target.row_end]
                )
                candidates = [
                    _NintLinearModule(
                        self._gpu_encoded_cache[(target.name, option.specs[target.name])],
                        bias,
                    )
                    for option in group.options
                ]
                if len(candidates) == 1:
                    parts.append(candidates[0])
                else:
                    parts.append(
                        _SoftNintLinearModule(
                            candidates,
                            logits[group.name],
                            temperature,
                        )
                    )
            replacement = parts[0] if len(parts) == 1 else _SplitSoftNintLinearModule(parts)
            _replace_module(layer, module_name, replacement)

        state = dict(self.index.layer_state(layer_index, exclude=excluded))
        result = layer.load_state_dict(state, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Qwen3.5 soft layer {layer_index} state mismatch: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        self.clear_quantized_cache(layer_index)
        return layer.to(device=self.device, dtype=self.soft_dtype).eval().requires_grad_(False)

    @contextmanager
    def layer(self, layer_index: int, *, quantized: bool) -> Iterator[nn.Module]:
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError(f"Qwen3.5 layer {layer_index} is out of range")
        layer = (
            self._load_quantized_layer(layer_index)
            if quantized
            else self._load_dense_layer(layer_index, self.teacher_dtype)
        )
        try:
            yield layer
        finally:
            del layer
            if quantized:
                self.clear_quantized_cache(layer_index)
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()

    @contextmanager
    def layer_for_strategy(
        self,
        layer_index: int,
        specs: Mapping[str, NintSpec],
    ) -> Iterator[nn.Module]:
        expected = {target.name for target in self.targets_by_layer[layer_index]}
        if set(specs) != expected:
            missing = sorted(expected - set(specs))
            extra = sorted(set(specs) - expected)
            raise ValueError(
                f"layer {layer_index} strategy tensor mismatch: "
                f"missing={missing[:4]}, extra={extra[:4]}"
            )
        layer = self._load_quantized_layer(layer_index, specs)
        try:
            yield layer
        finally:
            del layer
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()

    @contextmanager
    def layer_for_soft_assignment(
        self,
        layer_index: int,
        groups: Sequence[PrecisionGroup],
        logits: Mapping[str, torch.Tensor],
        temperature: float,
    ) -> Iterator[nn.Module]:
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError(f"Qwen3.5 layer {layer_index} is out of range")
        layer = self._load_soft_layer(layer_index, groups, logits, temperature)
        try:
            yield layer
        finally:
            del layer
            self.clear_quantized_cache(layer_index)
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()

    def _load_final_norm(self, dtype: torch.dtype) -> nn.Module:
        if self.is_moe:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeRMSNorm as RMSNorm,
            )
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5RMSNorm as RMSNorm,
            )

        with torch.device("meta"):
            norm = RMSNorm(
                self.hidden_size,
                eps=float(self.config.rms_norm_eps),
            )
        result = norm.load_state_dict(
            {
                "weight": self.index.tensor(
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
        return norm.to(device=self.device, dtype=dtype).eval().requires_grad_(False)

    @contextmanager
    def terminal_objective(self) -> Iterator[ChunkedTerminalObjective]:
        reference_norm = self._load_final_norm(self.teacher_dtype)
        candidate_norm = self._load_final_norm(self.quantized_dtype)
        head_name = (
            "lm_head.weight"
            if "lm_head.weight" in self.index.weight_map
            else "model.language_model.embed_tokens.weight"
        )
        shard = self.root / self.index.weight_map[head_name]
        try:
            with safe_open(str(shard), framework="pt", device="cpu") as reader:
                yield ChunkedTerminalObjective(
                    reference_norm,
                    candidate_norm,
                    _OpenSafetensorRows(reader, head_name),
                )
        finally:
            del reference_norm, candidate_norm
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()

    def clear_quantized_cache(self, layer_index: int | None = None) -> None:
        if layer_index is None:
            self._gpu_encoded_cache.clear()
        else:
            names = {target.name for target in self.targets_by_layer[layer_index]}
            self._gpu_encoded_cache = {
                key: value for key, value in self._gpu_encoded_cache.items() if key[0] not in names
            }
        gc.collect()

    def _forward_layer(
        self,
        layer: nn.Module,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        from transformers.masking_utils import create_causal_mask

        batch, sequence, _hidden = hidden_states.shape
        positions = torch.arange(sequence, device=self.device, dtype=torch.int64)
        text_positions = positions.unsqueeze(0).expand(batch, -1)
        rope_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
        position_embeddings = self.rotary(hidden_states, rope_positions)
        if self.config.layer_types[layer_index] == "full_attention":
            attention_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=None,
                cache_position=None,
                past_key_values=None,
                position_ids=text_positions,
            )
        else:
            attention_mask = None
        output = layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=text_positions,
            past_key_values=None,
            use_cache=False,
        )
        return output

    @torch.inference_mode()
    def forward_layer(
        self,
        layer: nn.Module,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_layer(layer, layer_index, hidden_states)

    def forward_layer_with_grad(
        self,
        layer: nn.Module,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_layer(layer, layer_index, hidden_states)


__all__ = ["Qwen35LayerwiseBackend"]
