"""Routed NINTM and NEPQ execution primitives for MLX."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's MLX runtime requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.formats.tpq import CccpPqTensor
from mfq.formats.moe import NintMoeTensor
from mfq.formats.nepq import NepqTensor
from mfq.formats.nint8_zero import Nint8ZeroTensor
from mfq.formats.npq0_l import Npq0LTensor
from mfq.formats.npq0_s import Npq0STensor
from mfq.formats.nvq import NvqJscTensor, NvqTensor
from mfq.formats.nvq1_l import Nvq1LTensor
from mfq.formats.nvq1_s import Nvq1STensor
from mfq.kernels.metal.tpq import (
    MetalCccpMoeWeight,
    MetalCccpPqWeight,
    cccp_grouped_moe_matmul,
    cccp_pq_routed_matmul,
)
from mfq.kernels.metal.kimi_k3 import situ_split
from mfq.kernels.metal.moe import (
    MetalMoeWeight,
    UnsupportedGroupedMoeError,
    grouped_moe_matmul,
)
from mfq.kernels.metal.moe_ops import (
    moe_topk,
    swiglu_split,
    weighted_reduce,
)
from mfq.kernels.metal.nint import MetalNintWeight, nint_matmul
from mfq.kernels.metal.nint8_zero import (
    MetalNint8ZeroWeight,
    nint8_zero_matmul,
)
from mfq.kernels.metal.vq import MetalVqWeight, vq_matmul
from mfq.quantize.nint_quant import NintTensor

_VQ_TYPES = (
    NvqTensor,
    NvqJscTensor,
    Nvq1LTensor,
    Nvq1STensor,
    Npq0LTensor,
    Npq0STensor,
    NepqTensor,
)


@dataclass(frozen=True)
class _MlxMoePool:
    expert_ids: mx.array
    weight: MetalNintWeight | MetalNint8ZeroWeight | MetalVqWeight | MetalCccpPqWeight
    experts: int
    out_per_expert: int

    def forward(
        self,
        x: mx.array,
        selected_ids: mx.array,
    ) -> mx.array:
        if isinstance(self.weight, MetalCccpPqWeight):
            return cccp_pq_routed_matmul(
                self.weight,
                x,
                selected_ids,
                self.expert_ids,
                out_per_expert=self.out_per_expert,
            )
        if isinstance(self.weight, MetalNintWeight):
            value = nint_matmul(self.weight, x)
        elif isinstance(self.weight, MetalNint8ZeroWeight):
            value = nint8_zero_matmul(self.weight, x)
        else:
            value = vq_matmul(self.weight, x)
        return value.reshape((*x.shape[:-1], self.experts, self.out_per_expert))


class MlxRoutedLinear:
    """Execute one NINTM tensor for explicit ``[token,route]`` expert IDs."""

    def __init__(
        self,
        tensor: NintMoeTensor,
        *,
        use_grouped: bool = True,
    ) -> None:
        self.n_experts = tensor.n_experts
        self.out_per_expert = tensor.out_per_expert
        self.neuron_len = tensor.neuron_len
        self.grouped_projection: int | None = None
        self.grouped_weight: MetalMoeWeight | MetalCccpMoeWeight | None = None
        has_cccp = any(isinstance(pool.tensor, CccpPqTensor) for pool in tensor.pools)
        all_cccp = all(isinstance(pool.tensor, CccpPqTensor) for pool in tensor.pools)
        if use_grouped:
            if all_cccp:
                self.grouped_weight = MetalCccpMoeWeight.from_tensor(tensor)
            elif not has_cccp:
                with suppress(UnsupportedGroupedMoeError):
                    self.grouped_weight = MetalMoeWeight.from_tensor(tensor)
        if self.grouped_weight is not None:
            self.pools = ()
            return

        pools: list[_MlxMoePool] = []
        for pool in tensor.pools:
            source = pool.tensor
            if isinstance(source, NintTensor):
                weight: (
                    MetalNintWeight | MetalNint8ZeroWeight | MetalVqWeight | MetalCccpPqWeight
                ) = MetalNintWeight.from_tensor(source)
            elif isinstance(source, Nint8ZeroTensor):
                weight = MetalNint8ZeroWeight.from_tensor(source)
            elif isinstance(source, _VQ_TYPES):
                weight = MetalVqWeight.from_tensor(source)
            elif isinstance(source, CccpPqTensor):
                weight = MetalCccpPqWeight.from_tensor(source)
            else:
                raise TypeError(
                    "Metal NINTM supports NINT/NVQ/NPQ/NEPQ/CCCP cohorts; "
                    f"received {type(source).__name__}"
                )
            expert_ids = np.ascontiguousarray(pool.expert_ids, dtype=np.int32)
            pools.append(
                _MlxMoePool(
                    expert_ids=mx.array(expert_ids),
                    weight=weight,
                    experts=int(expert_ids.size),
                    out_per_expert=self.out_per_expert,
                )
            )
        self.pools = tuple(pools)

    @classmethod
    def _from_grouped_projection(
        cls,
        weight: MetalMoeWeight,
        projection: int,
    ) -> MlxRoutedLinear:
        if not 0 <= int(projection) < weight.projections:
            raise ValueError("grouped projection index is out of range")
        result = object.__new__(cls)
        result.n_experts = weight.experts
        result.out_per_expert = weight.out_per_expert
        result.neuron_len = weight.neuron_len
        result.grouped_projection = int(projection)
        result.grouped_weight = weight
        result.pools = ()
        return result

    @property
    def uses_grouped_kernel(self) -> bool:
        """Whether routed matmul uses one heterogeneous Metal dispatch."""

        return self.grouped_weight is not None

    def forward(
        self,
        x: mx.array | np.ndarray,
        expert_ids: mx.array | np.ndarray,
    ) -> mx.array:
        if self.grouped_weight is not None:
            result = (
                cccp_grouped_moe_matmul(
                    self.grouped_weight,
                    x,
                    expert_ids,
                )
                if isinstance(self.grouped_weight, MetalCccpMoeWeight)
                else grouped_moe_matmul(self.grouped_weight, x, expert_ids)
            )
            if self.grouped_projection is not None:
                start = self.grouped_projection * self.out_per_expert
                result = result[..., start : start + self.out_per_expert]
            return result

        source = x if isinstance(x, mx.array) else mx.array(x)
        ids = expert_ids if isinstance(expert_ids, mx.array) else mx.array(expert_ids)
        if ids.dtype not in (mx.int32, mx.uint32):
            ids = ids.astype(mx.int32)
        if ids.ndim != 2:
            raise ValueError("routed expert IDs must have [tokens,routes] shape")
        tokens, routes = (int(item) for item in ids.shape)
        if source.ndim == 2:
            if tuple(int(item) for item in source.shape) != (tokens, self.neuron_len):
                raise ValueError("shared routed input must have [tokens,neuron_len] shape")
            source = mx.broadcast_to(
                source[:, None, :],
                (tokens, routes, self.neuron_len),
            )
        elif source.ndim != 3 or tuple(int(item) for item in source.shape) != (
            tokens,
            routes,
            self.neuron_len,
        ):
            raise ValueError("routed input must have [tokens,K] or [tokens,routes,K] shape")
        if source.dtype not in (mx.float16, mx.float32):
            source = source.astype(mx.float16)
        result = mx.zeros(
            (tokens, routes, self.out_per_expert),
            dtype=source.dtype,
        )
        for pool in self.pools:
            candidates = pool.forward(source, ids)
            if isinstance(pool.weight, MetalCccpPqWeight):
                selected = candidates
            else:
                membership = ids[:, :, None] == pool.expert_ids[None, None, :]
                selected = (candidates * membership[:, :, :, None].astype(candidates.dtype)).sum(
                    axis=2
                )
            result = result + selected
        return result

    def combine(
        self,
        x: mx.array | np.ndarray,
        expert_ids: mx.array | np.ndarray,
        route_weights: mx.array | np.ndarray,
    ) -> mx.array:
        values = self.forward(x, expert_ids)
        weights = route_weights if isinstance(route_weights, mx.array) else mx.array(route_weights)
        if tuple(int(item) for item in weights.shape) != tuple(
            int(item) for item in values.shape[:2]
        ):
            raise ValueError("route weights must have [tokens,routes] shape")
        return weighted_reduce(values, weights)

    def __call__(
        self,
        x: mx.array | np.ndarray,
        expert_ids: mx.array | np.ndarray,
    ) -> mx.array:
        return self.forward(x, expert_ids)


class MlxRoutedSwiGLUFFN:
    """Routed gate/up/down NINTM FFN with route-weighted reduction."""

    def __init__(
        self,
        gate: NintMoeTensor,
        up: NintMoeTensor,
        down: NintMoeTensor,
    ) -> None:
        gate_up_weight = MetalMoeWeight.concatenate_projections(
            (
                MetalMoeWeight.from_tensor(gate),
                MetalMoeWeight.from_tensor(up),
            )
        )
        self.gate_up_weight = gate_up_weight
        self.gate = MlxRoutedLinear._from_grouped_projection(gate_up_weight, 0)
        self.up = MlxRoutedLinear._from_grouped_projection(gate_up_weight, 1)
        self.down = MlxRoutedLinear(down)
        if (
            self.gate.n_experts != self.up.n_experts
            or self.gate.n_experts != self.down.n_experts
            or self.gate.out_per_expert != self.up.out_per_expert
            or self.gate.out_per_expert != self.down.neuron_len
            or self.gate.neuron_len != self.down.out_per_expert
        ):
            raise ValueError("routed SwiGLU gate/up/down shapes are incompatible")

    @property
    def uses_grouped_gate_up(self) -> bool:
        """Whether gate/up share one heterogeneous grouped matmul."""

        return self.gate_up_weight is not None

    def forward(
        self,
        x: mx.array | np.ndarray,
        expert_ids: mx.array | np.ndarray,
        route_weights: mx.array | np.ndarray,
    ) -> mx.array:
        gate_up = grouped_moe_matmul(self.gate_up_weight, x, expert_ids)
        hidden = swiglu_split(gate_up)
        return self.down.combine(hidden, expert_ids, route_weights)

    def forward_from_logits(
        self,
        x: mx.array | np.ndarray,
        router_logits: mx.array | np.ndarray,
        top_k: int,
        *,
        use_sigmoid: bool = False,
        use_sqrt_softplus: bool = False,
        normalize: bool = False,
        delayed_softmax: bool = False,
        router_bias: mx.array | np.ndarray | None = None,
        router_scale: float = 1.0,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Route and execute the complete SwiGLU FFN on Metal."""

        ids, weights = moe_topk(
            router_logits,
            top_k,
            use_sigmoid=use_sigmoid,
            use_sqrt_softplus=use_sqrt_softplus,
            normalize=normalize,
            delayed_softmax=delayed_softmax,
            bias=router_bias,
            scale=router_scale,
        )
        return self.forward(x, ids, weights), ids, weights

    def __call__(
        self,
        x: mx.array | np.ndarray,
        expert_ids: mx.array | np.ndarray,
        route_weights: mx.array | np.ndarray,
    ) -> mx.array:
        return self.forward(x, expert_ids, route_weights)


class MlxRoutedSiTUFFN:
    """TPQ2 combined gate/up and down expert FFN with SiTU activation."""

    def __init__(
        self,
        gate_up: NintMoeTensor,
        down: NintMoeTensor,
        *,
        beta: float,
        linear_beta: float | None,
    ) -> None:
        self.gate_up = MlxRoutedLinear(gate_up)
        self.down = MlxRoutedLinear(down)
        if (
            self.gate_up.n_experts != self.down.n_experts
            or self.gate_up.out_per_expert % 2
            or self.gate_up.out_per_expert // 2 != self.down.neuron_len
            or self.gate_up.neuron_len != self.down.out_per_expert
        ):
            raise ValueError("routed SiTU gate_up/down shapes are incompatible")
        self.beta = float(beta)
        self.linear_beta = None if linear_beta is None else float(linear_beta)

    def forward(
        self,
        x: mx.array | np.ndarray,
        expert_ids: mx.array | np.ndarray,
        route_weights: mx.array | np.ndarray,
    ) -> mx.array:
        gate_up = self.gate_up(x, expert_ids)
        hidden = situ_split(
            gate_up,
            beta=self.beta,
            linear_beta=self.linear_beta,
        )
        return self.down.combine(hidden, expert_ids, route_weights)

    def __call__(
        self,
        x: mx.array | np.ndarray,
        expert_ids: mx.array | np.ndarray,
        route_weights: mx.array | np.ndarray,
    ) -> mx.array:
        return self.forward(x, expert_ids, route_weights)


__all__ = [
    "MlxRoutedLinear",
    "MlxRoutedSiTUFFN",
    "MlxRoutedSwiGLUFFN",
]
