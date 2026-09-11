"""Architecture-aware BF16 activation-imatrix collection on CUDA and Metal."""

from __future__ import annotations

import gc
import hashlib
import json
import time
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _activation_kind(value: Any) -> str | None:
    if isinstance(value, str):
        name = value
    else:
        name = " ".join(
            (
                str(getattr(value, "__name__", "")),
                type(value).__name__,
                str(value),
            )
        )
    lowered = name.lower()
    if "silu" in lowered or "swish" in lowered:
        return "silu"
    if "gelu" in lowered:
        return "gelu"
    if "relu" in lowered:
        return "relu"
    if "sigmoid" in lowered:
        return "sigmoid"
    if "tanh" in lowered:
        return "tanh"
    return None


def _activation_derivative(kind: str, value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    if kind == "silu":
        probability = torch.sigmoid(value)
        return probability * (1.0 + value * (1.0 - probability))
    if kind == "gelu":
        inv_sqrt_two = 2.0**-0.5
        inv_sqrt_two_pi = (2.0 * torch.pi) ** -0.5
        return 0.5 * (1.0 + torch.erf(value * inv_sqrt_two)) + (
            value * torch.exp(-0.5 * value.square()) * inv_sqrt_two_pi
        )
    if kind == "relu":
        return (value > 0).to(value.dtype)
    if kind == "sigmoid":
        probability = torch.sigmoid(value)
        return probability * (1.0 - probability)
    if kind == "tanh":
        activated = torch.tanh(value)
        return 1.0 - activated.square()
    raise ValueError(f"unsupported NAQ activation derivative: {kind}")


def _normalized_naq_rows(coupled: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0:
        raise RuntimeError("NAQ target received no activations")
    value = coupled / float(count)
    mean = value.mean()
    if not torch.isfinite(mean) or float(mean) <= 1e-30:
        return torch.ones_like(value, dtype=torch.float32)
    value.div_(mean)
    if not torch.isfinite(value).all() or bool((value < 0).any()):
        raise FloatingPointError("NAQ importance contains invalid values")
    return value.to(torch.float32)


class _NaqBinding:
    category = "activation_corrected"

    def __init__(self, collector: ActivationImatrixCollector, target: ImatrixTarget) -> None:
        self.collector = collector
        self.target = target
        self.handles: list[Any] = []
        self.cache: dict[str, torch.Tensor] = {}

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.target.name,)

    def entry(self) -> ImportanceEntry | None:
        raise NotImplementedError

    def entries(self) -> dict[str, ImportanceEntry]:
        entry = self.entry()
        return {} if entry is None else {self.target.name: entry}

    def _standard(
        self, target: ImatrixTarget | None = None
    ) -> tuple[torch.Tensor, int]:
        selected = self.target if target is None else target
        count = int(self.collector.counts[selected.name][0].item())
        if count <= 0:
            return torch.empty(0, device=self.collector.device), 0
        standard = self.collector.sums[selected.name][0].float() / float(count)
        return standard, count


class _DenseFfnNaqBinding(_NaqBinding):
    category = "ffn_gate"

    def __init__(
        self,
        collector: ActivationImatrixCollector,
        gate_target: ImatrixTarget,
        up_target: ImatrixTarget,
        gate: nn.Module,
        up: nn.Module,
        down: nn.Module,
        activation: Any,
        activation_kind: str,
    ) -> None:
        super().__init__(collector, gate_target)
        self.up_target = up_target
        gate_weight = gate.weight
        down_weight = down.weight
        self.rows = int(gate_weight.shape[0])
        self.width = int(gate_weight.shape[1])
        self.activation = activation
        self.activation_kind = activation_kind
        self.gate_coupled = torch.zeros(
            self.rows,
            device=collector.device,
            dtype=collector.accumulation_dtype,
        )
        self.up_coupled = torch.zeros(
            self.rows,
            device=collector.device,
            dtype=collector.accumulation_dtype,
        )
        self.downstream_norm2 = down_weight.detach().float().square().sum(0)
        self.observations = 0
        self.handles.extend(
            (
                gate.register_forward_pre_hook(self._gate_pre),
                gate.register_forward_hook(self._gate_post),
                up.register_forward_hook(self._up_post),
                down.register_forward_pre_hook(self._down_pre),
            )
        )

    def _gate_pre(self, _module, inputs) -> None:
        self.cache["x"] = self.collector._matrix(inputs[0], self.width, self.target.name)

    def _gate_post(self, _module, _inputs, output) -> None:
        self.cache["gate"] = self.collector._matrix(output, self.rows, self.target.name)

    def _up_post(self, _module, _inputs, output) -> None:
        self.cache["up"] = self.collector._matrix(output, self.rows, self.target.name)

    def _down_pre(self, _module, _inputs) -> None:
        x = self.cache.pop("x")
        gate = self.cache.pop("gate")
        up = self.cache.pop("up")
        if gate.shape != up.shape or gate.shape[0] != x.shape[0]:
            raise RuntimeError(f"inconsistent NAQ FFN observations for {self.target.name}")
        gate_sensitivity = up.float() * _activation_derivative(
            self.activation_kind, gate
        )
        up_sensitivity = self.activation(gate).float()
        self.gate_coupled.add_(
            (gate_sensitivity.square() * self.downstream_norm2).sum(
                0, dtype=self.collector.accumulation_dtype
            )
        )
        self.up_coupled.add_(
            (up_sensitivity.square() * self.downstream_norm2).sum(
                0, dtype=self.collector.accumulation_dtype
            )
        )
        self.observations += int(x.shape[0])

    @property
    def names(self) -> tuple[str, ...]:
        return (self.target.name, self.up_target.name)

    def entries(self) -> dict[str, ImportanceEntry]:
        gate_standard, gate_count = self._standard(self.target)
        up_standard, up_count = self._standard(self.up_target)
        if gate_count <= 0 or up_count <= 0:
            return {}
        if (
            self.cache
            or self.observations != gate_count
            or self.observations != up_count
        ):
            raise RuntimeError(
                f"invalid NAQ FFN state for {self.target.name}: "
                f"observations={self.observations}, gate_count={gate_count}, "
                f"up_count={up_count}, cache={sorted(self.cache)}"
            )
        result = {}
        for target, standard, coupled in (
            (self.target, gate_standard, self.gate_coupled),
            (self.up_target, up_standard, self.up_coupled),
        ):
            row_importance = _normalized_naq_rows(coupled, gate_count)
            result[target.name] = ImportanceEntry(
                np.ascontiguousarray(
                    standard.detach().cpu().numpy()[None, :], dtype=np.float32
                ),
                np.asarray([gate_count], dtype=np.int64),
                np.ascontiguousarray(
                    row_importance.detach().cpu().numpy(), dtype=np.float32
                ),
            )
        return result


class _AttentionGateNaqBinding(_NaqBinding):
    category = "attention_gate"

    def __init__(
        self,
        collector: ActivationImatrixCollector,
        target: ImatrixTarget,
        query: nn.Module,
        output: nn.Module,
        head_dim: int,
    ) -> None:
        super().__init__(collector, target)
        query_weight = query.weight
        output_weight = output.weight
        self.rows = int(query_weight.shape[0])
        self.width = int(query_weight.shape[1])
        self.gate_rows = self.rows // 2
        heads = self.rows // (2 * head_dim)
        self.gate_indices = (
            torch.arange(heads, device=collector.device, dtype=torch.int64)[:, None]
            * (2 * head_dim)
            + head_dim
            + torch.arange(head_dim, device=collector.device, dtype=torch.int64)[None, :]
        ).reshape(-1)
        self.coupled = torch.zeros(
            self.gate_rows,
            device=collector.device,
            dtype=collector.accumulation_dtype,
        )
        self.downstream_norm2 = output_weight.detach().float().square().sum(0)
        self.observations = 0
        self.handles.extend(
            (
                query.register_forward_pre_hook(self._query_pre),
                query.register_forward_hook(self._query_post),
                output.register_forward_pre_hook(self._output_pre),
            )
        )

    def _query_pre(self, _module, inputs) -> None:
        self.cache["x"] = self.collector._matrix(inputs[0], self.width, self.target.name)

    def _query_post(self, _module, _inputs, value) -> None:
        projected = self.collector._matrix(value, self.rows, self.target.name)
        self.cache["gate"] = projected.index_select(1, self.gate_indices)

    def _output_pre(self, _module, inputs) -> None:
        x = self.cache.pop("x")
        gate = self.cache.pop("gate").float()
        gated_attention = self.collector._matrix(
            inputs[0], self.gate_rows, self.target.name
        ).float()
        if gate.shape != gated_attention.shape or gate.shape[0] != x.shape[0]:
            raise RuntimeError(f"inconsistent NAQ attention observations for {self.target.name}")
        sensitivity = gated_attention * (1.0 - torch.sigmoid(gate))
        self.coupled.add_(
            (sensitivity.square() * self.downstream_norm2).sum(
                0, dtype=self.collector.accumulation_dtype
            )
        )
        self.observations += int(x.shape[0])

    def entry(self) -> ImportanceEntry | None:
        standard, count = self._standard()
        if count <= 0:
            return None
        if self.cache or self.observations != count:
            raise RuntimeError(
                f"invalid NAQ attention state for {self.target.name}: "
                f"observations={self.observations}, count={count}, cache={sorted(self.cache)}"
            )
        row_importance = self.collector.neuron_mean(self.target)
        row_importance.index_copy_(
            0, self.gate_indices, self.coupled / float(count)
        )
        row_importance = _normalized_naq_rows(row_importance, 1)
        return ImportanceEntry(
            np.ascontiguousarray(standard.detach().cpu().numpy()[None, :], dtype=np.float32),
            np.asarray([count], dtype=np.int64),
            np.ascontiguousarray(row_importance.detach().cpu().numpy(), dtype=np.float32),
        )


class _GatedNormNaqBinding(_NaqBinding):
    category = "linear_attention_gate"

    def __init__(
        self,
        collector: ActivationImatrixCollector,
        target: ImatrixTarget,
        gate: nn.Module,
        norm: nn.Module,
        output: nn.Module,
        activation_kind: str,
    ) -> None:
        super().__init__(collector, target)
        gate_weight = gate.weight
        norm_weight = norm.weight
        output_weight = output.weight
        self.rows = int(gate_weight.shape[0])
        self.width = int(gate_weight.shape[1])
        self.head_dim = int(norm_weight.numel())
        self.epsilon = float(
            getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
        )
        self.norm_weight = norm_weight.detach().float().reshape(1, self.head_dim)
        self.activation_kind = activation_kind
        self.coupled = torch.zeros(
            self.rows,
            device=collector.device,
            dtype=collector.accumulation_dtype,
        )
        self.downstream_norm2 = output_weight.detach().float().square().sum(0)
        self.observations = 0
        self.handles.extend(
            (
                gate.register_forward_pre_hook(self._gate_pre),
                norm.register_forward_pre_hook(self._norm_pre),
            )
        )

    def _gate_pre(self, _module, inputs) -> None:
        self.cache["x"] = self.collector._matrix(inputs[0], self.width, self.target.name)

    def _norm_pre(self, _module, inputs) -> None:
        if len(inputs) < 2:
            raise TypeError("NAQ gated norm requires hidden and gate inputs")
        x = self.cache.pop("x")
        hidden = inputs[0].detach().reshape(-1, self.head_dim).float()
        gate = inputs[1].detach().reshape(-1, self.head_dim).float()
        variance = hidden.square().mean(-1, keepdim=True)
        normalized = hidden * torch.rsqrt(variance + self.epsilon)
        normalized.mul_(self.norm_weight)
        sensitivity = normalized * _activation_derivative(self.activation_kind, gate)
        sensitivity = self.collector._matrix(
            sensitivity.reshape(-1, self.rows), self.rows, self.target.name
        )
        if sensitivity.shape[0] != x.shape[0]:
            raise RuntimeError(f"inconsistent NAQ gated-norm observations for {self.target.name}")
        self.coupled.add_(
            (sensitivity.square() * self.downstream_norm2).sum(
                0, dtype=self.collector.accumulation_dtype
            )
        )
        self.observations += int(x.shape[0])

    def entry(self) -> ImportanceEntry | None:
        standard, count = self._standard()
        if count <= 0:
            return None
        if self.cache or self.observations != count:
            raise RuntimeError(
                f"invalid NAQ gated-norm state for {self.target.name}: "
                f"observations={self.observations}, count={count}, cache={sorted(self.cache)}"
            )
        row_importance = _normalized_naq_rows(self.coupled, count)
        return ImportanceEntry(
            np.ascontiguousarray(standard.detach().cpu().numpy()[None, :], dtype=np.float32),
            np.asarray([count], dtype=np.int64),
            np.ascontiguousarray(row_importance.detach().cpu().numpy(), dtype=np.float32),
        )


class ActivationImatrixCollector:
    """Collect factorized input-channel and output-neuron NAQ importance."""

    def __init__(
        self,
        targets: Sequence[ImatrixTarget],
        device: torch.device,
        *,
        accumulation_dtype: torch.dtype = torch.float64,
        neural: bool = False,
    ) -> None:
        if accumulation_dtype not in {torch.float32, torch.float64}:
            raise ValueError("imatrix accumulation dtype must be float32 or float64")
        self.targets = tuple(targets)
        self.device = device
        self.accumulation_dtype = accumulation_dtype
        self.neural = bool(neural)
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
        self.neuron_sums: dict[str, torch.Tensor] = {}
        self._active_naq: list[_NaqBinding] = []
        self._naq_entries: dict[str, ImportanceEntry] = {}
        self._naq_categories: dict[str, str] = {}
        self._expert_naq_rows: dict[str, torch.Tensor] = {}

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

    def add_neurons(self, target: ImatrixTarget, value: torch.Tensor) -> None:
        sums = self.neuron_sums[target.name]
        if sums.ndim != 1:
            raise RuntimeError(f"NAQ neuron shape is not linear for {target.name}")
        matrix = self._matrix(value, int(sums.shape[0]), target.name)
        sums.add_(matrix.square().sum(0, dtype=self.accumulation_dtype))

    def add_expert_neurons(
        self,
        target: ImatrixTarget,
        value: torch.Tensor,
        expert: int,
    ) -> None:
        sums = self.neuron_sums[target.name]
        if sums.ndim != 2:
            raise RuntimeError(f"NAQ neuron shape is not routed for {target.name}")
        matrix = self._matrix(value, int(sums.shape[1]), target.name)
        sums[expert].add_(matrix.square().sum(0, dtype=self.accumulation_dtype))

    def neuron_mean(self, target: ImatrixTarget) -> torch.Tensor:
        sums = self.neuron_sums.get(target.name)
        count = int(self.counts[target.name][0].item())
        if sums is None or sums.ndim != 1 or count <= 0:
            raise RuntimeError(f"NAQ neuron importance is unavailable for {target.name}")
        value = sums / float(count)
        if not torch.isfinite(value).all() or bool((value < 0).any()):
            raise FloatingPointError(f"NAQ neuron importance is invalid for {target.name}")
        return value

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
                if self.neural:
                    weight = getattr(module, "weight", None)
                    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                        raise TypeError(
                            f"NAQ target is not a matrix projection: {module_name}"
                        )
                    self.neuron_sums.setdefault(
                        target.name,
                        torch.zeros(
                            int(weight.shape[0]),
                            device=self.device,
                            dtype=self.accumulation_dtype,
                        ),
                    )

                    def post_hook(_module, _inputs, output, *, _target=target):
                        self.add_neurons(_target, output)

                    self.handles.append(module.register_forward_hook(post_hook))
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
            activation_kind = _activation_kind(activation) if self.neural else None
            if self.neural:
                self.neuron_sums.setdefault(
                    gate.name,
                    torch.zeros(
                        (gate.experts, int(gate_up.shape[1])),
                        device=self.device,
                        dtype=self.accumulation_dtype,
                    ),
                )
                self.neuron_sums.setdefault(
                    down.name,
                    torch.zeros(
                        (down.experts, int(down_proj.shape[1])),
                        device=self.device,
                        dtype=self.accumulation_dtype,
                    ),
                )
            if activation_kind is not None:
                rows_per_expert = int(gate_up.shape[1])
                self._expert_naq_rows.setdefault(
                    gate.name,
                    torch.zeros(
                        (gate.experts, rows_per_expert),
                        device=self.device,
                        dtype=self.accumulation_dtype,
                    ),
                )
                self._naq_categories[gate.name] = "routed_ffn_gate_up"

            def expert_forward(
                _module,
                hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                top_k_weights: torch.Tensor,
                *,
                _gate=gate,
                _down=down,
                _activation_kind=activation_kind,
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
                    if self.neural:
                        measured_gate_up = torch.cat((gate_value, up_value), dim=-1)
                        if valid is not None:
                            measured_gate_up = measured_gate_up[valid]
                        if measured_gate_up.numel():
                            self.add_expert_neurons(_gate, measured_gate_up, expert)
                    activated = _module.act_fn(gate_value)
                    intermediate = activated * up_value
                    if _activation_kind is not None and measured_current.numel():
                        measured_gate = gate_value if valid is None else gate_value[valid]
                        measured_up = up_value if valid is None else up_value[valid]
                        measured_activated = activated if valid is None else activated[valid]
                        measured_weight = top_k_weights[token_idx, top_k_pos]
                        if valid is not None:
                            measured_weight = measured_weight[valid]
                        derivative = _activation_derivative(
                            _activation_kind, measured_gate
                        )
                        downstream = (
                            _module.down_proj[expert]
                            .detach()
                            .float()
                            .square()
                            .sum(0)
                        )
                        route_weight2 = measured_weight.detach().float().square()[:, None]
                        gate_importance = (
                            (measured_up.float() * derivative).square()
                            * downstream
                            * route_weight2
                        ).sum(0, dtype=self.accumulation_dtype)
                        up_importance = (
                            measured_activated.float().square()
                            * downstream
                            * route_weight2
                        ).sum(0, dtype=self.accumulation_dtype)
                        self._expert_naq_rows[_gate.name][expert].add_(
                            torch.cat((gate_importance, up_importance))
                        )
                    measured_intermediate = intermediate if valid is None else intermediate[valid]
                    if measured_intermediate.numel():
                        self.add_expert(_down, measured_intermediate, expert)
                    output = torch.nn.functional.linear(
                        intermediate, _module.down_proj[expert]
                    )
                    output = output * top_k_weights[token_idx, top_k_pos, None]
                    if self.neural:
                        measured_output = output if valid is None else output[valid]
                        if measured_output.numel():
                            self.add_expert_neurons(_down, measured_output, expert)
                    final.index_add_(0, token_idx, output.to(final.dtype))
                return final

            self.restores.append((module, original))
            module.forward = types.MethodType(expert_forward, module)

        if self.neural:
            self._install_activation_corrections(modules, targets)

    @staticmethod
    def _child_name(parent: str, child: str) -> str:
        return f"{parent}.{child}" if parent else child

    def _install_activation_corrections(
        self,
        modules: Mapping[str, nn.Module],
        targets: Sequence[ImatrixTarget],
    ) -> None:
        linear_targets = {
            target.module_name: target for target in targets if target.kind == "linear"
        }
        bound = {
            name for binding in self._active_naq for name in binding.names
        }
        for parent_name, parent in modules.items():
            gate = getattr(parent, "gate_proj", None)
            up = getattr(parent, "up_proj", None)
            down = getattr(parent, "down_proj", None)
            activation = getattr(parent, "act_fn", None)
            target = linear_targets.get(self._child_name(parent_name, "gate_proj"))
            up_target = linear_targets.get(self._child_name(parent_name, "up_proj"))
            activation_kind = _activation_kind(activation)
            if (
                target is not None
                and up_target is not None
                and target.name not in bound
                and up_target.name not in bound
                and isinstance(gate, nn.Module)
                and isinstance(up, nn.Module)
                and isinstance(down, nn.Module)
                and activation_kind is not None
            ):
                gate_weight = getattr(gate, "weight", None)
                up_weight = getattr(up, "weight", None)
                down_weight = getattr(down, "weight", None)
                if (
                    isinstance(gate_weight, torch.Tensor)
                    and isinstance(up_weight, torch.Tensor)
                    and isinstance(down_weight, torch.Tensor)
                    and gate_weight.ndim == up_weight.ndim == down_weight.ndim == 2
                    and gate_weight.shape == up_weight.shape
                    and int(down_weight.shape[1]) == int(gate_weight.shape[0])
                ):
                    self._active_naq.append(
                        _DenseFfnNaqBinding(
                            self,
                            target,
                            up_target,
                            gate,
                            up,
                            down,
                            activation,
                            activation_kind,
                        )
                    )
                    bound.add(target.name)
                    bound.add(up_target.name)

            query = getattr(parent, "q_proj", None)
            output = getattr(parent, "o_proj", None)
            target = linear_targets.get(self._child_name(parent_name, "q_proj"))
            head_dim = int(getattr(parent, "head_dim", 0))
            if (
                target is not None
                and target.name not in bound
                and isinstance(query, nn.Module)
                and isinstance(output, nn.Module)
                and head_dim > 0
            ):
                query_weight = getattr(query, "weight", None)
                output_weight = getattr(output, "weight", None)
                if (
                    isinstance(query_weight, torch.Tensor)
                    and isinstance(output_weight, torch.Tensor)
                    and query_weight.ndim == output_weight.ndim == 2
                    and int(query_weight.shape[0]) == 2 * int(output_weight.shape[1])
                    and int(query_weight.shape[0]) % (2 * head_dim) == 0
                ):
                    self._active_naq.append(
                        _AttentionGateNaqBinding(
                            self, target, query, output, head_dim
                        )
                    )
                    bound.add(target.name)

            z = getattr(parent, "in_proj_z", None)
            norm = getattr(parent, "norm", None)
            output = getattr(parent, "out_proj", None)
            target = linear_targets.get(self._child_name(parent_name, "in_proj_z"))
            activation = getattr(parent, "act", getattr(parent, "activation", None))
            activation_kind = _activation_kind(activation)
            if (
                target is not None
                and target.name not in bound
                and isinstance(z, nn.Module)
                and isinstance(norm, nn.Module)
                and isinstance(output, nn.Module)
                and activation_kind is not None
            ):
                z_weight = getattr(z, "weight", None)
                norm_weight = getattr(norm, "weight", None)
                output_weight = getattr(output, "weight", None)
                if (
                    isinstance(z_weight, torch.Tensor)
                    and isinstance(norm_weight, torch.Tensor)
                    and isinstance(output_weight, torch.Tensor)
                    and z_weight.ndim == output_weight.ndim == 2
                    and int(z_weight.shape[0]) == int(output_weight.shape[1])
                    and int(norm_weight.numel()) > 0
                    and int(z_weight.shape[0]) % int(norm_weight.numel()) == 0
                ):
                    self._active_naq.append(
                        _GatedNormNaqBinding(
                            self, target, z, norm, output, activation_kind
                        )
                    )
                    bound.add(target.name)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for module, original in reversed(self.restores):
            module.forward = original
        self.restores.clear()
        bindings, self._active_naq = self._active_naq, []
        for binding in bindings:
            binding.remove()
        for binding in bindings:
            for name, entry in binding.entries().items():
                self._naq_entries[name] = entry
                self._naq_categories[name] = binding.category

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
        for name, sums in self.neuron_sums.items():
            ordinary = result[name]
            if sums.ndim == 1:
                row_importance = _normalized_naq_rows(
                    sums, int(ordinary.counts[0])
                ).detach().cpu().numpy()
                category = "linear"
            else:
                row_importance = np.ones(tuple(sums.shape), dtype=np.float32)
                for expert, count in enumerate(ordinary.counts):
                    if count > 0:
                        row_importance[expert] = (
                            _normalized_naq_rows(sums[expert], int(count))
                            .detach()
                            .cpu()
                            .numpy()
                        )
                category = "routed_linear"
            result[name] = ImportanceEntry(
                ordinary.values,
                ordinary.counts,
                np.ascontiguousarray(row_importance, dtype=np.float32).reshape(-1),
            )
            self._naq_categories.setdefault(name, category)
        for name, sums in self._expert_naq_rows.items():
            ordinary = result[name]
            row_importance = np.ones(tuple(sums.shape), dtype=np.float32)
            for expert, count in enumerate(ordinary.counts):
                if count <= 0:
                    continue
                corrected = sums[expert].detach().cpu().numpy().astype(np.float64)
                corrected /= float(count)
                mean = float(corrected.mean())
                if np.isfinite(mean) and mean > 1e-30:
                    corrected /= mean
                    row_importance[expert] = corrected.astype(np.float32)
            result[name] = ImportanceEntry(
                ordinary.values,
                ordinary.counts,
                np.ascontiguousarray(row_importance.reshape(-1)),
            )
        result.update(self._naq_entries)
        return result

    @property
    def naq_categories(self) -> dict[str, str]:
        return dict(self._naq_categories)


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
    # Conditional-generation checkpoints expose the language model through a
    # ``*_text`` sub-config, while text-only checkpoints use the family name
    # directly.  Backend selection is a property of that family, not of the
    # outer multimodal wrapper.
    backend_family = model_type.removesuffix("_text")
    if backend_family == "gemma4":
        from mfq.calibration.layerwise_gemma4 import Gemma4LayerwiseBackend

        return Gemma4LayerwiseBackend(model, device=device, attention=attention), model_type
    if backend_family in {"qwen3_5", "qwen3_5_moe"}:
        from mfq.calibration.layerwise_qwen35 import Qwen35LayerwiseBackend

        return (
            Qwen35LayerwiseBackend(
                model, None, device=device, quant_backend="cpu", attention=attention
            ),
            model_type,
        )
    raise ValueError(f"generic imatrix does not yet support model type {model_type!r}")


def _targets(
    backend: Any,
    _model_type: str,
) -> tuple[dict[int, tuple[ImatrixTarget, ...]], tuple[ImatrixTarget, ...]]:
    by_layer: dict[int, list[ImatrixTarget]] = {}
    index = backend.index
    for layer in range(backend.num_layers):
        prefix = f"model.language_model.layers.{layer}."
        values: list[ImatrixTarget] = []
        layer_names = tuple(
            name for name in sorted(index.weight_map) if name.startswith(prefix)
        )
        for name in layer_names:
            shape = index.shape(name)
            suffix = name[len(prefix) :]
            if len(shape) == 2 and suffix.endswith(".weight"):
                module_name = suffix.removesuffix(".weight")
                values.append(ImatrixTarget(name, module_name, int(shape[1])))
        # Discover fused routed experts from tensor rank and module structure,
        # rather than binding the generic collector to architecture names.
        for gate_name in layer_names:
            gate_shape = index.shape(gate_name)
            gate_suffix = gate_name[len(prefix) :].removesuffix(".weight")
            if len(gate_shape) != 3 or not gate_suffix.endswith(".gate_up_proj"):
                continue
            expert_module = gate_suffix.removesuffix(".gate_up_proj")
            down_candidates = (
                prefix + expert_module + ".down_proj",
                prefix + expert_module + ".down_proj.weight",
            )
            down_name = next(
                (candidate for candidate in down_candidates if candidate in index.weight_map),
                None,
            )
            if down_name is None:
                continue
            down_shape = index.shape(down_name)
            if (
                len(down_shape) != 3
                or int(gate_shape[0]) != int(down_shape[0])
                or int(gate_shape[1]) != 2 * int(down_shape[2])
                or int(gate_shape[2]) != int(down_shape[1])
            ):
                raise ValueError(
                    f"incompatible routed-expert tensors: {gate_name} {gate_shape}, "
                    f"{down_name} {down_shape}"
                )
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
    objective: str = "naq",
) -> ImportanceMatrix:
    """Collect one frozen train-only BF16 imatrix layer by layer."""

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
    if objective not in {"naq", "linear"}:
        raise ValueError("imatrix objective must be naq or linear")
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
    collector = ActivationImatrixCollector(
        targets,
        target_device,
        accumulation_dtype=dtype,
        neural=objective == "naq",
    )
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
                key_rows = int(backend.config.linear_num_key_heads) * int(
                    backend.config.linear_key_head_dim
                )
                value_rows = int(backend.config.linear_num_value_heads) * int(
                    backend.config.linear_value_head_dim
                )
                qk_rows = 2 * key_rows
                if (
                    entry.row_importance is not None
                    and entry.row_importance.size != qk_rows + value_rows
                ):
                    raise ValueError(
                        f"linear-attention NAQ neuron shape mismatch for {name}"
                    )
                entries[base + "in_proj_qk.weight"] = ImportanceEntry(
                    entry.values,
                    entry.counts,
                    (
                        None
                        if entry.row_importance is None
                        else np.ascontiguousarray(entry.row_importance[:qk_rows])
                    ),
                )
                entries[base + "in_proj_v.weight"] = ImportanceEntry(
                    entry.values,
                    entry.counts,
                    (
                        None
                        if entry.row_importance is None
                        else np.ascontiguousarray(
                            entry.row_importance[qk_rows : qk_rows + value_rows]
                        )
                    ),
                )
        metadata = {
            "objective": (
                "neuron_aware_factorized_importance"
                if objective == "naq"
                else "mean_squared_linear_input_activation"
            ),
            "ordinary_objective": "mean_squared_linear_input_activation",
            "naq_entries": collector.naq_categories,
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
