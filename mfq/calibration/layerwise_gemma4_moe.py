"""Differentiable expert-precision backend for Gemma4 MoE calibration."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mfq.calibration.artifact import CalibrationScheme
from mfq.calibration.layerwise_gemma4 import Gemma4LayerwiseBackend
from mfq.calibration.moe_soft_refinement import (
    CoupledExpertPrecisionProblem,
    _group_expert,
)
from mfq.calibration.rate_distortion import PrecisionGroup
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import NintTensor
from mfq.quantize.nint_quant import quantize as quantize_nint_cpu
from mfq.quantize.nint_quant_torch import quantize_axis0 as quantize_nint_cuda
from mfq.runtime.torch_linear import TorchNintLinear


def _active_experts(
    top_k_index: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        mask = torch.nn.functional.one_hot(top_k_index, num_classes=num_experts)
        mask = mask.permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten()
    return mask, hit


class _PackedGemma4Experts(nn.Module):
    """Hard selected experts evaluated through packed NINT CUDA matmul."""

    def __init__(
        self,
        gate_up: Sequence[TorchNintLinear],
        down: Sequence[TorchNintLinear],
        *,
        hidden_activation: str,
    ) -> None:
        super().__init__()
        if not gate_up or len(gate_up) != len(down):
            raise ValueError("hard Gemma4 experts require matching gate/down weights")
        from transformers.activations import ACT2FN

        self.num_experts = len(gate_up)
        self.gate_up = tuple(item.shared_weights_clone() for item in gate_up)
        self.down = tuple(item.shared_weights_clone() for item in down)
        self.act_fn = ACT2FN[hidden_activation]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        expert_mask, expert_hit = _active_experts(top_k_index, self.num_experts)
        for expert_tensor in expert_hit:
            expert = int(expert_tensor.item())
            top_k_pos, token_idx = torch.where(expert_mask[expert])
            current = hidden_states[token_idx]
            gate, up = self.gate_up[expert](current).chunk(2, dim=-1)
            output = self.down[expert](self.act_fn(gate) * up)
            output = output * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, output.to(final.dtype))
        return final


class _SoftGemma4Experts(nn.Module):
    """Exact mixture over complete quantized expert functions."""

    def __init__(
        self,
        gate_weights: torch.Tensor,
        down_weights: torch.Tensor,
        logits: Sequence[torch.Tensor],
        *,
        temperature: float,
        hidden_activation: str,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("soft Gemma4 expert temperature must be positive")
        if gate_weights.ndim != 4 or down_weights.ndim != 4:
            raise ValueError("soft Gemma4 candidate weights must be [E,C,O,K]")
        if gate_weights.shape[:2] != down_weights.shape[:2]:
            raise ValueError("soft Gemma4 gate/down expert counts disagree")
        self.num_experts = int(gate_weights.shape[0])
        self.candidate_count = int(gate_weights.shape[1])
        if self.candidate_count < 2:
            raise ValueError("soft Gemma4 experts require at least two candidates")
        if len(logits) != self.num_experts:
            raise ValueError("soft Gemma4 experts require one logit vector per expert")
        for value in logits:
            if value.shape != (self.candidate_count,):
                raise ValueError("soft Gemma4 expert logits disagree with candidate count")
        if gate_weights.shape[2] % 2:
            raise ValueError("soft Gemma4 gate/up output width must be even")
        if gate_weights.shape[2] // 2 != down_weights.shape[3]:
            raise ValueError("soft Gemma4 gate/up and down intermediate widths disagree")
        if gate_weights.shape[3] != down_weights.shape[2]:
            raise ValueError("soft Gemma4 gate/up and down hidden widths disagree")
        self.register_buffer("gate_weights", gate_weights, persistent=False)
        self.register_buffer("down_weights", down_weights, persistent=False)
        self._logits = tuple(logits)
        self.temperature = float(temperature)
        from transformers.activations import ACT2FN

        self.act_fn = ACT2FN[hidden_activation]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("soft Gemma4 experts require [tokens, hidden] input")
        tokens, hidden = hidden_states.shape
        if top_k_index.shape != top_k_weights.shape or top_k_index.shape[0] != tokens:
            raise ValueError("soft Gemma4 routing tensors have invalid shapes")

        routes = int(top_k_index.numel())
        flat_experts = top_k_index.reshape(-1).to(dtype=torch.int64)
        token_ids = (
            torch.arange(tokens, device=hidden_states.device, dtype=torch.int64)
            .unsqueeze(1)
            .expand_as(top_k_index)
            .reshape(-1)
        )
        order = torch.argsort(flat_experts, stable=True)
        sorted_experts = flat_experts[order]
        sorted_tokens = token_ids[order]
        sorted_route_weights = top_k_weights.reshape(-1)[order]
        counts = torch.bincount(flat_experts, minlength=self.num_experts)
        maximum_routes = int(counts.max().item())
        starts = torch.cumsum(counts, dim=0) - counts
        slots = (
            torch.arange(routes, device=hidden_states.device, dtype=torch.int64)
            - starts[sorted_experts]
        )

        expert_input = torch.zeros(
            (self.num_experts, maximum_routes, hidden),
            device=hidden_states.device,
            dtype=self.gate_weights.dtype,
        )
        expert_input[sorted_experts, slots] = hidden_states[sorted_tokens].to(
            dtype=expert_input.dtype
        )
        candidate_input = (
            expert_input[:, None]
            .expand(-1, self.candidate_count, -1, -1)
            .reshape(self.num_experts * self.candidate_count, maximum_routes, hidden)
        )
        gate_weight = self.gate_weights.reshape(
            self.num_experts * self.candidate_count,
            self.gate_weights.shape[2],
            hidden,
        )
        projected = torch.bmm(candidate_input, gate_weight.transpose(1, 2))
        gate, up = projected.chunk(2, dim=-1)
        intermediate = self.act_fn(gate) * up
        down_weight = self.down_weights.reshape(
            self.num_experts * self.candidate_count,
            hidden,
            self.down_weights.shape[3],
        )
        candidate_output = torch.bmm(intermediate, down_weight.transpose(1, 2))
        candidate_output = candidate_output.reshape(
            self.num_experts,
            self.candidate_count,
            maximum_routes,
            hidden,
        )
        probabilities = torch.stack(
            [
                torch.softmax(value / self.temperature, dim=0)
                for value in self._logits
            ],
            dim=0,
        )
        mixed = torch.einsum(
            "ec,ecmh->emh",
            probabilities,
            candidate_output.float(),
        )
        routed = mixed[sorted_experts, slots]
        routed = routed * sorted_route_weights[:, None]
        final = torch.zeros_like(hidden_states)
        final.index_add_(0, sorted_tokens, routed.to(final.dtype))
        return final


class Gemma4MoESoftBackend(Gemma4LayerwiseBackend):
    """Gemma4 backend that optimizes one coupled precision profile per expert."""

    soft_dtype = torch.bfloat16
    soft_min_autograd_rows = 1

    def __init__(
        self,
        model_path: str | Path,
        scheme: CalibrationScheme,
        problem: CoupledExpertPrecisionProblem,
        *,
        device: str | torch.device = "cuda:0",
        quant_backend: str = "cuda",
        attention: str = "sdpa",
        candidate_cache_dir: str | Path | None = None,
        soft_weight_cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__(model_path, device=device, attention=attention)
        if quant_backend not in {"cuda", "cpu"}:
            raise ValueError("quant_backend must be cuda or cpu")
        if quant_backend == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA quantization requested without CUDA support")
        self.scheme = scheme
        self.problem = problem
        self.quant_backend = quant_backend
        self.candidate_cache_dir = (
            None if candidate_cache_dir is None else Path(candidate_cache_dir).resolve()
        )
        if self.candidate_cache_dir is not None:
            self.candidate_cache_dir.mkdir(parents=True, exist_ok=True)
        self.soft_weight_cache_dir = (
            (
                self.candidate_cache_dir / "soft-bf16-layers"
                if self.candidate_cache_dir is not None
                else None
            )
            if soft_weight_cache_dir is None
            else Path(soft_weight_cache_dir).resolve()
        )
        if self.soft_weight_cache_dir is not None:
            self.soft_weight_cache_dir.mkdir(parents=True, exist_ok=True)
        if set(problem.tensors_by_layer) != set(range(self.num_layers)):
            raise ValueError("Gemma4 MoE candidates do not cover every decoder layer")
        if {group.layer for group in problem.groups} != set(range(self.num_layers)):
            raise ValueError("Gemma4 MoE precision groups do not cover every decoder layer")
        for layer, (gate_name, down_name) in problem.tensors_by_layer.items():
            gate_shape = self.index.shape(gate_name)
            down_shape = self.index.shape(down_name)
            gate_selection = scheme.require_expert(gate_name)
            down_selection = scheme.require_expert(down_name)
            expected_gate = (
                gate_selection.n_experts,
                gate_selection.rows_per_expert,
                gate_selection.columns,
            )
            expected_down = (
                down_selection.n_experts,
                down_selection.rows_per_expert,
                down_selection.columns,
            )
            if gate_shape != expected_gate or down_shape != expected_down:
                raise ValueError(
                    f"Gemma4 layer {layer} expert shapes disagree: "
                    f"{gate_shape}/{down_shape} vs {expected_gate}/{expected_down}"
                )

    def _quantize(self, weight: torch.Tensor, spec: NintSpec) -> NintTensor:
        if self.quant_backend == "cuda":
            return quantize_nint_cuda(weight, spec, device=self.device)
        return quantize_nint_cpu(weight.float().cpu().numpy(), spec, axis=0)

    def _cache_path(
        self,
        source_name: str,
        expert: int,
        spec: NintSpec,
    ) -> Path | None:
        if self.candidate_cache_dir is None:
            return None
        shard = self.root / self.index.weight_map[source_name]
        stat = shard.stat()
        identity = "|".join(
            (
                str(self.root),
                source_name,
                str(expert),
                str(spec.bits),
                str(spec.groupsize),
                str(spec.sub_bits),
                str(stat.st_size),
                str(stat.st_mtime_ns),
            )
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self.candidate_cache_dir / f"{digest}.gemma4-expert.nint-exec.npz"

    def _load_cached_linear(
        self,
        source_name: str,
        expert: int,
        spec: NintSpec,
    ) -> TorchNintLinear | None:
        path = self._cache_path(source_name, expert, spec)
        if path is None or not path.is_file():
            return None
        shape = self.index.shape(source_name)[1:]
        with np.load(path, allow_pickle=False) as archive:
            document = json.loads(archive["metadata"].tobytes().decode("utf-8"))
            expected = {
                "source_name": source_name,
                "expert": expert,
                "bits": spec.bits,
                "groupsize": spec.groupsize,
                "sub_bits": spec.sub_bits,
                "shape": list(shape),
                "axis": 0,
                "neuron_len": shape[1],
            }
            if document != expected:
                raise ValueError(f"invalid Gemma4 expert candidate cache: {path}")
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
            shape=shape,
            axis=0,
            neuron_len=shape[1],
            device=self.device,
        )

    def _save_cached_linear(
        self,
        source_name: str,
        expert: int,
        encoded: NintTensor,
    ) -> None:
        path = self._cache_path(source_name, expert, encoded.spec)
        if path is None or path.exists():
            return
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        if temporary.exists():
            raise FileExistsError(f"candidate cache temporary already exists: {temporary}")
        metadata = np.frombuffer(
            json.dumps(
                {
                    "source_name": source_name,
                    "expert": expert,
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

    def _linear(
        self,
        source_name: str,
        expert: int,
        spec: NintSpec,
    ) -> TorchNintLinear:
        cached = self._load_cached_linear(source_name, expert, spec)
        if cached is not None:
            return cached
        source = self.index.tensor(
            source_name,
            row_start=expert,
            row_end=expert + 1,
            device="cpu",
        ).squeeze(0)
        encoded = self._quantize(source, spec)
        self._save_cached_linear(source_name, expert, encoded)
        result = TorchNintLinear(encoded, self.device)
        del source, encoded
        return result

    def _layer_groups(
        self,
        layer_index: int,
        groups: Sequence[PrecisionGroup],
    ) -> tuple[PrecisionGroup, ...]:
        layer_groups = tuple(
            sorted(
                (group for group in groups if group.layer == layer_index),
                key=_group_expert,
            )
        )
        expected = self.scheme.require_expert(
            self.problem.tensors_by_layer[layer_index][0]
        ).n_experts
        if len(layer_groups) != expected:
            raise ValueError(
                f"Gemma4 layer {layer_index} has {len(layer_groups)} soft groups; "
                f"expected {expected}"
            )
        if tuple(_group_expert(group) for group in layer_groups) != tuple(range(expected)):
            raise ValueError(f"Gemma4 layer {layer_index} soft groups have missing expert IDs")
        return layer_groups

    def _soft_weight_cache_paths(
        self,
        layer_index: int,
        option_profiles: Sequence[str],
    ) -> tuple[Path, Path, dict[str, object]] | None:
        if self.soft_weight_cache_dir is None:
            return None
        gate_name, down_name = self.problem.tensors_by_layer[layer_index]
        sources = []
        for name in (gate_name, down_name):
            shard = self.root / self.index.weight_map[name]
            stat = shard.stat()
            sources.append(
                {
                    "name": name,
                    "shape": list(self.index.shape(name)),
                    "shard": str(shard),
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        document: dict[str, object] = {
            "format": "mfq.gemma4-soft-bf16-layer-cache.v1",
            "model": str(self.root),
            "layer": layer_index,
            "candidate_sha256": self.problem.candidate_sha256,
            "profiles": list(option_profiles),
            "dtype": "bfloat16",
            "sources": sources,
        }
        encoded = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        stem = self.soft_weight_cache_dir / f"layer-{layer_index:03d}-{digest}"
        return stem.with_suffix(".bf16"), stem.with_suffix(".json"), document

    @staticmethod
    def _weight_cache_elements(
        gate_shape: Sequence[int],
        down_shape: Sequence[int],
        candidate_count: int,
    ) -> int:
        return candidate_count * (
            math.prod(int(value) for value in gate_shape)
            + math.prod(int(value) for value in down_shape)
        )

    def _read_soft_weight_cache(
        self,
        data_path: Path,
        metadata_path: Path,
        expected_metadata: Mapping[str, object],
        gate_shape: Sequence[int],
        down_shape: Sequence[int],
        candidate_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not data_path.exists() and not metadata_path.exists():
            return None
        if not data_path.is_file() or not metadata_path.is_file():
            raise ValueError(f"incomplete Gemma4 soft weight cache: {data_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata != dict(expected_metadata):
            raise ValueError(f"Gemma4 soft weight cache metadata changed: {metadata_path}")
        expected_elements = self._weight_cache_elements(
            gate_shape,
            down_shape,
            candidate_count,
        )
        expected_bytes = expected_elements * torch.empty(
            (), dtype=self.soft_dtype
        ).element_size()
        if data_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"Gemma4 soft weight cache has {data_path.stat().st_size} bytes; "
                f"expected {expected_bytes}"
            )

        gate = torch.empty(
            (gate_shape[0], candidate_count, *gate_shape[1:]),
            device=self.device,
            dtype=self.soft_dtype,
        )
        down = torch.empty(
            (down_shape[0], candidate_count, *down_shape[1:]),
            device=self.device,
            dtype=self.soft_dtype,
        )
        chunk_elements = 32 << 20
        host = torch.empty(
            chunk_elements,
            dtype=self.soft_dtype,
            device="cpu",
            pin_memory=self.device.type == "cuda",
        )
        with data_path.open("rb", buffering=0) as stream:
            for destination in (gate.reshape(-1), down.reshape(-1)):
                offset = 0
                while offset < destination.numel():
                    count = min(chunk_elements, destination.numel() - offset)
                    byte_view = host[:count].view(torch.uint8).numpy()
                    filled = 0
                    while filled < byte_view.size:
                        read = stream.readinto(byte_view[filled:])
                        if not read:
                            raise EOFError(f"truncated Gemma4 soft weight cache: {data_path}")
                        filled += read
                    destination[offset : offset + count].copy_(
                        host[:count],
                        non_blocking=False,
                    )
                    offset += count
            if stream.read(1):
                raise ValueError(f"Gemma4 soft weight cache has a trailing payload: {data_path}")
        return gate, down

    def _save_soft_weight_cache(
        self,
        data_path: Path,
        metadata_path: Path,
        metadata: Mapping[str, object],
        gate: torch.Tensor,
        down: torch.Tensor,
    ) -> None:
        if data_path.exists() or metadata_path.exists():
            raise FileExistsError(f"Gemma4 soft weight cache already exists: {data_path}")
        data_temporary = data_path.with_suffix(data_path.suffix + f".{os.getpid()}.tmp")
        metadata_temporary = metadata_path.with_suffix(
            metadata_path.suffix + f".{os.getpid()}.tmp"
        )
        chunk_elements = 32 << 20
        try:
            with data_temporary.open("xb", buffering=0) as stream:
                for source in (gate.reshape(-1), down.reshape(-1)):
                    for start in range(0, source.numel(), chunk_elements):
                        end = min(start + chunk_elements, source.numel())
                        host = source[start:end].to(device="cpu", non_blocking=False)
                        view = memoryview(host.view(torch.uint8).numpy())
                        written = 0
                        while written < len(view):
                            count = stream.write(view[written:])
                            if not count:
                                raise OSError(
                                    f"failed to write Gemma4 soft weight cache: {data_path}"
                                )
                            written += count
            metadata_temporary.write_text(
                json.dumps(dict(metadata), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(data_temporary, data_path)
            os.replace(metadata_temporary, metadata_path)
        except BaseException:
            data_temporary.unlink(missing_ok=True)
            metadata_temporary.unlink(missing_ok=True)
            raise

    def _soft_experts(
        self,
        layer_index: int,
        groups: Sequence[PrecisionGroup],
        logits: Mapping[str, torch.Tensor],
        temperature: float,
    ) -> _SoftGemma4Experts:
        layer_groups = self._layer_groups(layer_index, groups)
        option_profiles = tuple(option.profile for option in layer_groups[0].options)
        for group in layer_groups:
            if tuple(option.profile for option in group.options) != option_profiles:
                raise ValueError("Gemma4 experts disagree on candidate profile ordering")
            if group.name not in logits:
                raise ValueError(f"missing soft logits for {group.name}")
        gate_name, down_name = self.problem.tensors_by_layer[layer_index]
        gate_shape = self.index.shape(gate_name)
        down_shape = self.index.shape(down_name)
        cache = self._soft_weight_cache_paths(layer_index, option_profiles)
        if cache is not None:
            data_path, metadata_path, metadata = cache
            cached = self._read_soft_weight_cache(
                data_path,
                metadata_path,
                metadata,
                gate_shape,
                down_shape,
                len(option_profiles),
            )
            if cached is not None:
                gate_candidates, down_candidates = cached
                return _SoftGemma4Experts(
                    gate_candidates,
                    down_candidates,
                    [logits[group.name] for group in layer_groups],
                    temperature=temperature,
                    hidden_activation=str(self.config.hidden_activation),
                )
        gate_candidates = torch.empty(
            (gate_shape[0], len(option_profiles), *gate_shape[1:]),
            device=self.device,
            dtype=self.soft_dtype,
        )
        down_candidates = torch.empty(
            (down_shape[0], len(option_profiles), *down_shape[1:]),
            device=self.device,
            dtype=self.soft_dtype,
        )
        for candidate, profile in enumerate(option_profiles):
            for expert in range(gate_shape[0]):
                record = self.problem.candidate(layer_index, expert, profile)
                gate_linear = self._linear(gate_name, expert, record.spec)
                down_linear = self._linear(down_name, expert, record.spec)
                gate_candidates[expert, candidate].copy_(
                    gate_linear.weight.to(dtype=self.soft_dtype)
                )
                down_candidates[expert, candidate].copy_(
                    down_linear.weight.to(dtype=self.soft_dtype)
                )
                del gate_linear, down_linear
        if cache is not None:
            self._save_soft_weight_cache(
                data_path,
                metadata_path,
                metadata,
                gate_candidates,
                down_candidates,
            )
        return _SoftGemma4Experts(
            gate_candidates,
            down_candidates,
            [logits[group.name] for group in layer_groups],
            temperature=temperature,
            hidden_activation=str(self.config.hidden_activation),
        )

    def _hard_experts(
        self,
        layer_index: int,
        groups: Sequence[PrecisionGroup],
        profiles: Mapping[str, str],
    ) -> _PackedGemma4Experts:
        layer_groups = self._layer_groups(layer_index, groups)
        if any(group.name not in profiles for group in layer_groups):
            raise ValueError(f"Gemma4 layer {layer_index} hard profiles are incomplete")
        gate_name, down_name = self.problem.tensors_by_layer[layer_index]
        gate: list[TorchNintLinear] = []
        down: list[TorchNintLinear] = []
        for expert, group in enumerate(layer_groups):
            record = self.problem.candidate(
                layer_index,
                expert,
                profiles[group.name],
            )
            gate.append(self._linear(gate_name, expert, record.spec))
            down.append(self._linear(down_name, expert, record.spec))
        return _PackedGemma4Experts(
            gate,
            down,
            hidden_activation=str(self.config.hidden_activation),
        )

    def _load_layer_with_experts(
        self,
        layer_index: int,
        experts: nn.Module,
        *,
        dtype: torch.dtype,
    ) -> nn.Module:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

        with torch.device("meta"):
            layer = Gemma4TextDecoderLayer(self.config, layer_index)
        layer.experts = experts
        gate_name, down_name = self.problem.tensors_by_layer[layer_index]
        state = dict(
            self.index.layer_state(
                layer_index,
                exclude={gate_name, down_name},
            )
        )
        result = layer.load_state_dict(state, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Gemma4 soft layer {layer_index} state mismatch: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        return layer.to(device=self.device, dtype=dtype).eval().requires_grad_(False)

    @contextmanager
    def layer_for_soft_assignment(
        self,
        layer_index: int,
        groups: Sequence[PrecisionGroup],
        logits: Mapping[str, torch.Tensor],
        temperature: float,
    ) -> Iterator[nn.Module]:
        experts = self._soft_experts(layer_index, groups, logits, temperature)
        layer = self._load_layer_with_experts(
            layer_index,
            experts,
            dtype=self.soft_dtype,
        )
        try:
            yield layer
        finally:
            del layer, experts
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
        groups = self._layer_groups(layer_index, self.problem.groups)
        profiles: dict[str, str] = {}
        expected_names = {name for group in groups for name in group.tensor_names}
        if set(specs) != expected_names:
            raise ValueError(f"Gemma4 layer {layer_index} strategy tensor mismatch")
        for group in groups:
            matches = [
                option.profile
                for option in group.options
                if all(specs[name] == option.specs[name] for name in group.tensor_names)
            ]
            if len(matches) != 1:
                raise ValueError(f"Gemma4 group {group.name} has no unique hard profile")
            profiles[group.name] = matches[0]
        experts = self._hard_experts(layer_index, groups, profiles)
        layer = self._load_layer_with_experts(
            layer_index,
            experts,
            dtype=self.quantized_dtype,
        )
        try:
            yield layer
        finally:
            del layer, experts
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()

    def clear_quantized_cache(self, layer_index: int | None = None) -> None:
        # Candidate weights live only for the active layer context. The packed
        # persistent cache is on disk and immutable.
        gc.collect()

    def forward_layer_with_grad(
        self,
        layer: nn.Module,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_layer(layer, layer_index, hidden_states)

    def _forward_layer(
        self,
        layer: nn.Module,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Keep the same masks, RoPE and shared-MoE ordering as the BF16 backend.
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )

        batch, sequence, _hidden = hidden_states.shape
        position_ids = torch.arange(
            sequence,
            device=self.device,
            dtype=torch.int64,
        ).unsqueeze(0).expand(batch, -1)
        layer_type = str(self.config.layer_types[layer_index])
        position_embeddings = self.rotary(hidden_states, position_ids, layer_type)
        mask_fn = (
            create_causal_mask
            if layer_type == "full_attention"
            else create_sliding_window_causal_mask
        )
        attention_mask = mask_fn(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        )
        return layer(
            hidden_states,
            shared_kv_states={},
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )

    @torch.inference_mode()
    def forward_layer(
        self,
        layer: nn.Module,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_layer(layer, layer_index, hidden_states)


__all__ = ["Gemma4MoESoftBackend"]
