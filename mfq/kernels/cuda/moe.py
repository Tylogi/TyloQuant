"""CUDA MoE routing and expert-wise NINT ``mul_mat_id`` primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch

from mfq.kernels.cuda._ext import ext
from mfq.kernels.cuda.activation import silu_mul


def _hetero_profile_code(bits: int, gs: int) -> int:
    return {
        (2, 16): 6,
        (4, 24): 0,
        (5, 28): 1,
        (6, 24): 2,
        (8, 48): 3,
        (8, 24): 4,
    }.get((int(bits), int(gs)), -1)


@dataclass
class NintExpertPool:
    """One homogeneous execution cohort inside an expert-wise weight tensor.

    ``expert_ids[local]`` owns rows
    ``[local * out_per_expert:(local + 1) * out_per_expert]`` in ``weight``.
    """

    weight: dict
    expert_ids: tuple[int, ...]
    _local_maps: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def local_map(self, n_experts: int, device: torch.device) -> torch.Tensor:
        key = str(device)
        cached = self._local_maps.get(key)
        if cached is not None and cached.device == device:
            return cached
        mapping = torch.full((n_experts,), -1, dtype=torch.int32)
        if self.expert_ids:
            global_ids = torch.tensor(self.expert_ids, dtype=torch.int64)
            mapping[global_ids] = torch.arange(len(self.expert_ids), dtype=torch.int32)
        cached = mapping.to(device=device, non_blocking=True)
        self._local_maps[key] = cached
        return cached


@dataclass
class CompactExpertPool:
    """One native NINT, NVQ/NPQ, or NEPQ execution cohort."""

    family: str
    weight: dict
    expert_ids: tuple[int, ...]
    _local_maps: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def local_map(self, n_experts: int, device: torch.device) -> torch.Tensor:
        key = str(device)
        cached = self._local_maps.get(key)
        if cached is not None and cached.device == device:
            return cached
        mapping = torch.full((n_experts,), -1, dtype=torch.int32)
        if self.expert_ids:
            global_ids = torch.tensor(self.expert_ids, dtype=torch.int64)
            mapping[global_ids] = torch.arange(len(self.expert_ids), dtype=torch.int32)
        cached = mapping.to(device=device, non_blocking=True)
        self._local_maps[key] = cached
        return cached


@dataclass
class ExpertWiseNintWeight:
    """A logical ``[experts, out, in]`` tensor split into precision cohorts."""

    n_experts: int
    out_per_expert: int
    neuron_len: int
    pools: tuple[NintExpertPool, ...]
    _activation_workspaces: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False
    )
    _hetero_metadata: dict[str, tuple[torch.Tensor, ...]] = field(
        default_factory=dict, init=False, repr=False
    )
    _hetero_workspaces: dict[tuple, tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.n_experts <= 0 or self.out_per_expert <= 0 or self.neuron_len <= 0:
            raise ValueError("expert-wise NINT dimensions must be positive")
        if not self.pools:
            raise ValueError("expert-wise NINT weight must contain at least one pool")
        owners = [-1] * self.n_experts
        for pool_index, pool in enumerate(self.pools):
            if not pool.expert_ids:
                raise ValueError("expert-wise NINT pool cannot be empty")
            g = pool.weight
            expected_rows = len(pool.expert_ids) * self.out_per_expert
            if int(g["out"]) != expected_rows:
                raise ValueError(
                    f"pool {pool_index} has {g['out']} rows; expected {expected_rows}"
                )
            if int(g["neuron_len"]) != self.neuron_len:
                raise ValueError(f"pool {pool_index} input width mismatch")
            bits = int(g.get("bits", 4))
            if bits not in (2, 3, 4, 5, 6, 8):
                raise ValueError(f"pool {pool_index} uses unsupported NINT{bits}")
            for expert in pool.expert_ids:
                if not 0 <= expert < self.n_experts:
                    raise ValueError(f"expert id {expert} is outside [0, {self.n_experts})")
                if owners[expert] != -1:
                    raise ValueError(f"expert {expert} appears in more than one precision pool")
                owners[expert] = pool_index
        missing = [expert for expert, owner in enumerate(owners) if owner < 0]
        if missing:
            raise ValueError(f"expert-wise NINT pools do not cover experts {missing[:16]}")

    @classmethod
    def homogeneous(
        cls,
        weight: dict,
        n_experts: int,
        out_per_expert: int | None = None,
    ) -> "ExpertWiseNintWeight":
        if out_per_expert is None:
            if int(weight["out"]) % n_experts:
                raise ValueError("flattened expert rows are not divisible by n_experts")
            out_per_expert = int(weight["out"]) // n_experts
        return cls(
            n_experts=int(n_experts),
            out_per_expert=int(out_per_expert),
            neuron_len=int(weight["neuron_len"]),
            pools=(NintExpertPool(weight, tuple(range(int(n_experts)))),),
        )

    def activation_workspace(
        self,
        x: torch.Tensor,
        *,
        gs: int,
        groups: int,
        input_rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_pad = groups * gs
        key = (str(x.device), input_rows, k_pad, groups)
        cached = self._activation_workspaces.get(key)
        if cached is not None and cached[0].device == x.device:
            return cached
        cached = (
            torch.empty((input_rows, k_pad), device=x.device, dtype=torch.int8),
            torch.empty((input_rows, groups), device=x.device, dtype=torch.float32),
        )
        self._activation_workspaces[key] = cached
        return cached

    @property
    def hetero_supported(self) -> bool:
        return all(
            _hetero_profile_code(int(pool.weight.get("bits", 4)), int(pool.weight["gs"])) >= 0
            for pool in self.pools
        )

    def hetero_metadata(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        key = str(device)
        cached = self._hetero_metadata.get(key)
        if cached is not None:
            return cached
        weight_ptrs: list[list[int]] = []
        pool_params: list[list[int]] = []
        expert_pool = [-1] * self.n_experts
        expert_local = [-1] * self.n_experts
        for pool_index, pool in enumerate(self.pools):
            weight = pool.weight
            tensors = (
                weight["q_packed"],
                weight["sub_scale"],
                weight["sub_min"],
                weight["neuron_scale"],
                weight["neuron_min"],
            )
            if any(tensor.device != device for tensor in tensors):
                raise ValueError("heterogeneous expert metadata and input must share a device")
            weight_ptrs.append([int(tensor.data_ptr()) for tensor in tensors])
            profile = _hetero_profile_code(int(weight.get("bits", 4)), int(weight["gs"]))
            if profile < 0:
                raise ValueError("unsupported heterogeneous expert profile")
            pool_params.append([profile, int(weight["ng"])])
            for local, expert in enumerate(pool.expert_ids):
                expert_pool[expert] = pool_index
                expert_local[expert] = local
        cached = (
            torch.tensor(weight_ptrs, device=device, dtype=torch.int64),
            torch.tensor(pool_params, device=device, dtype=torch.int32),
            torch.tensor(expert_pool, device=device, dtype=torch.int32),
            torch.tensor(expert_local, device=device, dtype=torch.int32),
        )
        self._hetero_metadata[key] = cached
        return cached

    def hetero_workspace(
        self, x: torch.Tensor, input_rows: int
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor]:
        profile_shapes = tuple((int(pool.weight["gs"]), int(pool.weight["ng"])) for pool in self.pools)
        key = (str(x.device), int(input_rows), profile_shapes)
        cached = self._hetero_workspaces.get(key)
        if cached is not None:
            return cached
        qx_list: list[torch.Tensor] = []
        xscale_list: list[torch.Tensor] = []
        pointers: list[list[int]] = []
        for pool, (gs, groups) in zip(self.pools, profile_shapes, strict=True):
            qx, xscale = self.activation_workspace(
                x, gs=gs, groups=groups, input_rows=input_rows
            )
            qx_list.append(qx)
            xscale_list.append(xscale)
            pointers.append([int(qx.data_ptr()), int(xscale.data_ptr())])
        cached = (
            tuple(qx_list),
            tuple(xscale_list),
            torch.tensor(pointers, device=x.device, dtype=torch.int64),
        )
        self._hetero_workspaces[key] = cached
        return cached


@dataclass
class ExpertWiseMixedWeight:
    """A logical expert tensor dispatched by native precision-family cohorts."""

    n_experts: int
    out_per_expert: int
    neuron_len: int
    pools: tuple[CompactExpertPool, ...]
    _activation_workspaces: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.n_experts <= 0 or self.out_per_expert <= 0 or self.neuron_len <= 0:
            raise ValueError("expert-wise mixed dimensions must be positive")
        if not self.pools:
            raise ValueError("expert-wise mixed weight must contain at least one pool")
        owners = [-1] * self.n_experts
        for pool_index, pool in enumerate(self.pools):
            if pool.family not in {"nint", "nvq", "nepq"}:
                raise ValueError(f"unsupported expert cohort family: {pool.family}")
            if not pool.expert_ids:
                raise ValueError("expert-wise mixed pool cannot be empty")
            g = pool.weight
            if int(g["neuron_len"]) != self.neuron_len:
                raise ValueError(f"pool {pool_index} input width mismatch")
            if pool.family == "nepq":
                if (
                    int(g["n_experts"]) != len(pool.expert_ids)
                    or int(g["out_per_expert"]) != self.out_per_expert
                ):
                    raise ValueError(f"NEPQ pool {pool_index} shape mismatch")
            else:
                expected_rows = len(pool.expert_ids) * self.out_per_expert
                if int(g["out"]) != expected_rows:
                    raise ValueError(
                        f"pool {pool_index} has {g['out']} rows; expected {expected_rows}"
                    )
            for expert in pool.expert_ids:
                if not 0 <= expert < self.n_experts:
                    raise ValueError(f"expert id {expert} is outside [0, {self.n_experts})")
                if owners[expert] != -1:
                    raise ValueError(f"expert {expert} appears in more than one precision pool")
                owners[expert] = pool_index
        missing = [expert for expert, owner in enumerate(owners) if owner < 0]
        if missing:
            raise ValueError(f"expert-wise mixed pools do not cover experts {missing[:16]}")

    def activation_workspace(
        self,
        x: torch.Tensor,
        *,
        gs: int,
        groups: int,
        input_rows: int,
        transform_key: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_pad = groups * gs
        key = (str(x.device), input_rows, k_pad, groups, transform_key)
        cached = self._activation_workspaces.get(key)
        if cached is not None and cached[0].device == x.device:
            return cached
        cached = (
            torch.empty((input_rows, k_pad), device=x.device, dtype=torch.int8),
            torch.empty((input_rows, groups), device=x.device, dtype=torch.float32),
        )
        self._activation_workspaces[key] = cached
        return cached


@dataclass(frozen=True)
class MoeRoutePlan:
    """GPU route metadata shared by gate, up, and down expert projections."""

    ids: torch.Tensor
    n_experts: int
    ids_dst: torch.Tensor
    expert_bounds: torch.Tensor
    tile_bounds: torch.Tensor
    tile_experts: torch.Tensor
    counts: torch.Tensor
    cursors: torch.Tensor

    @property
    def tokens(self) -> int:
        return int(self.ids.shape[0])

    @property
    def routes(self) -> int:
        return int(self.ids.shape[1])

    @property
    def map_ready(self) -> bool:
        return self.tokens <= 8 or self.ids_dst.numel() == self.ids.numel()

    @classmethod
    def build(cls, ids: torch.Tensor, n_experts: int) -> "MoeRoutePlan":
        if ids.ndim != 2:
            raise ValueError("ids must have [tokens, routes] shape")
        ids = ids.contiguous().to(device=ids.device, dtype=torch.int32)
        if int(ids.shape[0]) > 8:
            ids_dst, expert_bounds, tile_bounds, tile_experts, counts = ext().moe_build_expert_map_cuda(
                ids, int(n_experts), 8
            )
            cursors = torch.empty(0, device=ids.device, dtype=torch.int32)
        else:
            empty = torch.empty(0, device=ids.device, dtype=torch.int32)
            ids_dst = empty
            expert_bounds = empty
            tile_bounds = empty
            tile_experts = empty
            counts = empty
            cursors = empty
        return cls(
            ids=ids,
            n_experts=int(n_experts),
            ids_dst=ids_dst,
            expert_bounds=expert_bounds,
            tile_bounds=tile_bounds,
            tile_experts=tile_experts,
            counts=counts,
            cursors=cursors,
        )


def topk(
    logits: torch.Tensor,
    top_k: int,
    *,
    use_sigmoid: bool = False,
    use_sqrt_softplus: bool = False,
    normalize: bool = False,
    delayed_softmax: bool = False,
    bias: torch.Tensor | None = None,
    norm_floor: float = 1e-20,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the router transform and return contiguous int32 ids/f32 weights."""

    logits = logits.reshape(-1, logits.shape[-1]).contiguous()
    if bias is not None:
        bias = bias.contiguous().to(device=logits.device, dtype=torch.float32)
    ids, weights = ext().moe_topk_cuda(
        logits,
        int(top_k),
        bool(use_sigmoid),
        bool(use_sqrt_softplus),
        bool(normalize),
        bool(delayed_softmax),
        bias,
        float(norm_floor),
        float(scale),
    )
    return ids, weights


def sqrtsoftplus_weights(
    logits: torch.Tensor,
    ids: torch.Tensor,
    *,
    norm_floor: float = 1e-20,
    scale: float = 1.0,
) -> torch.Tensor:
    """Gather hash-selected router weights using DeepSeek V4's transform."""

    logits = logits.reshape(-1, logits.shape[-1]).contiguous()
    ids = ids.reshape(logits.shape[0], -1).contiguous().to(
        device=logits.device, dtype=torch.int32
    )
    return ext().moe_sqrtsoftplus_weights_cuda(
        logits,
        ids,
        float(norm_floor),
        float(scale),
    )


ExpertWiseWeight = ExpertWiseNintWeight | ExpertWiseMixedWeight


def _prepare_input(weight: ExpertWiseWeight, x: torch.Tensor) -> torch.Tensor:
    if x.ndim not in (2, 3):
        raise ValueError("expert input must have [T,K] or [T,R,K] shape")
    x = x.contiguous().to(torch.float16)
    if x.shape[-1] > weight.neuron_len:
        raise ValueError(f"input width {x.shape[-1]} exceeds {weight.neuron_len}")
    if x.shape[-1] < weight.neuron_len:
        x = torch.nn.functional.pad(x, (0, weight.neuron_len - x.shape[-1]))
    return x


def _grouped_matmul_mixed(
    weight: ExpertWiseMixedWeight,
    x: torch.Tensor,
    route: MoeRoutePlan,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    from mfq.kernels.cuda.nepq_matmul import (
        _prepare_routed_input as prepare_nepq_routed_input,
    )
    from mfq.kernels.cuda.nepq_matmul import nepq_grouped_matmul_pool
    from mfq.kernels.cuda.nvq_matmul import nvq_grouped_matmul_pool

    if route.n_experts != weight.n_experts:
        raise ValueError("route and weight expert counts differ")
    x = _prepare_input(weight, x)
    if x.shape[0] != route.tokens:
        raise ValueError("input and route token counts differ")
    if x.ndim == 3 and x.shape[1] != route.routes:
        raise ValueError("routed input and route counts differ")
    if x.device != route.ids.device:
        raise ValueError("input and route tensors must be on the same device")
    expected = (route.tokens, route.routes, weight.out_per_expert)
    if out is None:
        out = torch.empty(expected, device=x.device, dtype=torch.float16)
    elif (
        tuple(out.shape) != expected
        or out.dtype != torch.float16
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be contiguous float16 {expected} on {x.device}")

    input_rows = route.tokens * route.routes if x.ndim == 3 else route.tokens
    prepared_inputs: dict[tuple, torch.Tensor] = {("identity",): x}
    quantized: set[tuple] = set()
    for pool in weight.pools:
        g = pool.weight
        if pool.family == "nepq" and int(g.get("rotation_block", 0)):
            transform_key = (
                "hadamard",
                int(g["rotation_block"]),
                int(g.get("rotation_seed", 0)),
            )
            value = prepared_inputs.get(transform_key)
            if value is None:
                value = prepare_nepq_routed_input(g, x)
                prepared_inputs[transform_key] = value
        else:
            transform_key = ("identity",)
            value = x
        gs = int(g["gs"])
        groups = int(g["ng"])
        activation_key = (transform_key, gs, groups)
        qx, xscale = weight.activation_workspace(
            value,
            gs=gs,
            groups=groups,
            input_rows=input_rows,
            transform_key=transform_key,
        )
        input_quantized = activation_key in quantized
        expert_local = pool.local_map(weight.n_experts, x.device)
        if pool.family == "nint":
            ext().nint_moe_grouped_matmul_pool_ws_cuda(
                g["q_packed"],
                g["sub_scale"],
                g["sub_min"],
                g["neuron_scale"],
                g["neuron_min"],
                value,
                route.ids,
                expert_local,
                weight.n_experts,
                len(pool.expert_ids),
                weight.out_per_expert,
                gs,
                int(g.get("bits", 4)),
                route.map_ready,
                input_quantized,
                out,
                qx,
                xscale,
                route.counts,
                route.cursors,
                route.ids_dst,
                route.expert_bounds,
                route.tile_bounds,
                route.tile_experts,
            )
        elif pool.family == "nvq":
            nvq_grouped_matmul_pool(
                g,
                value,
                route,
                expert_local,
                len(pool.expert_ids),
                out=out,
                qx=qx,
                xscale=xscale,
                input_quantized=input_quantized,
            )
        else:
            nepq_grouped_matmul_pool(
                g,
                value,
                route,
                expert_local,
                len(pool.expert_ids),
                out=out,
                qx=qx,
                xscale=xscale,
                input_quantized=input_quantized,
                input_prepared=True,
            )
        quantized.add(activation_key)
    return out


def grouped_matmul(
    weight: ExpertWiseWeight,
    x: torch.Tensor,
    route: MoeRoutePlan,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run expert-wise NINT ``mul_mat_id`` and return ``[T,R,out]`` pairs."""

    if isinstance(weight, ExpertWiseMixedWeight):
        return _grouped_matmul_mixed(weight, x, route, out=out)
    if route.n_experts != weight.n_experts:
        raise ValueError("route and weight expert counts differ")
    x = _prepare_input(weight, x)
    if x.shape[0] != route.tokens:
        raise ValueError("input and route token counts differ")
    if x.ndim == 3 and x.shape[1] != route.routes:
        raise ValueError("routed input and route counts differ")
    if x.device != route.ids.device:
        raise ValueError("input and route tensors must be on the same device")
    if out is None:
        out = torch.empty(
            (route.tokens, route.routes, weight.out_per_expert),
            device=x.device,
            dtype=torch.float16,
        )
    else:
        expected = (route.tokens, route.routes, weight.out_per_expert)
        if tuple(out.shape) != expected or out.dtype != torch.float16 or out.device != x.device:
            raise ValueError(f"out must be float16 {expected} on {x.device}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    input_rows = route.tokens * route.routes if x.ndim == 3 else route.tokens
    if weight.hetero_supported and route.tokens <= 2:
        weight_ptrs, pool_params, expert_pool, expert_local = weight.hetero_metadata(x.device)
        qx_list, xscale_list, activation_ptrs = weight.hetero_workspace(x, input_rows)
        quantized_groups: set[tuple[int, int]] = set()
        for pool, qx, xscale in zip(weight.pools, qx_list, xscale_list, strict=True):
            gs = int(pool.weight["gs"])
            groups = int(pool.weight["ng"])
            key = (gs, groups)
            if key in quantized_groups:
                continue
            ext().nint_moe_quantize_input_ws_cuda(x, gs, qx, xscale)
            quantized_groups.add(key)
        profile_mask = 0
        for pool in weight.pools:
            profile_mask |= 1 << _hetero_profile_code(
                int(pool.weight.get("bits", 4)), int(pool.weight["gs"])
            )
        return ext().nint_moe_grouped_matmul_hetero_qx_cuda(
            weight_ptrs,
            pool_params,
            activation_ptrs,
            expert_pool,
            expert_local,
            route.ids,
            profile_mask,
            weight.n_experts,
            weight.out_per_expert,
            weight.neuron_len,
            x.ndim == 3,
            out,
            route.ids_dst,
            route.expert_bounds,
            route.tile_bounds,
            route.tile_experts,
        )
    quantized_groups: set[tuple[int, int]] = set()
    for pool in weight.pools:
        g = pool.weight
        gs = int(g["gs"])
        groups = int(g["ng"])
        key = (gs, groups)
        qx, xscale = weight.activation_workspace(
            x, gs=gs, groups=groups, input_rows=input_rows
        )
        ext().nint_moe_grouped_matmul_pool_ws_cuda(
            g["q_packed"],
            g["sub_scale"],
            g["sub_min"],
            g["neuron_scale"],
            g["neuron_min"],
            x,
            route.ids,
            pool.local_map(weight.n_experts, x.device),
            weight.n_experts,
            len(pool.expert_ids),
            weight.out_per_expert,
            gs,
            int(g.get("bits", 4)),
            route.map_ready,
            key in quantized_groups,
            out,
            qx,
            xscale,
            route.counts,
            route.cursors,
            route.ids_dst,
            route.expert_bounds,
            route.tile_bounds,
            route.tile_experts,
        )
        quantized_groups.add(key)
    return out


def weighted_reduce(pair_output: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Reduce ``[T,R,O]`` expert outputs with router weights in FP32."""

    return ext().moe_weighted_reduce_cuda(
        pair_output.contiguous().to(torch.float16),
        weights.contiguous().to(device=pair_output.device, dtype=torch.float32),
    )


def swiglu_split(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply SwiGLU to contiguous ``[..., 2 * width]`` expert output."""

    return ext().moe_swiglu_split_cuda(gate_up.contiguous().to(torch.float16))


def geglu_split(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply tanh-approximate GeGLU to contiguous ``[..., 2 * width]`` output."""

    return ext().moe_geglu_split_cuda(gate_up.contiguous().to(torch.float16))


def add_shared_gate(
    routed: torch.Tensor,
    shared: torch.Tensor,
    gate_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute ``routed + sigmoid(gate_logits) * shared`` in one CUDA kernel."""

    return ext().moe_add_shared_gate_cuda(
        routed.contiguous().to(torch.float16),
        shared.contiguous().to(device=routed.device, dtype=torch.float16),
        gate_logits.reshape(-1, 1).contiguous().to(device=routed.device, dtype=torch.float32),
    )


def weighted_reduce_shared_gate(
    pair_output: torch.Tensor,
    weights: torch.Tensor,
    shared: torch.Tensor,
    gate_logits: torch.Tensor,
) -> torch.Tensor:
    """Reduce routed outputs and add the gated shared expert in one kernel."""

    return ext().moe_weighted_reduce_shared_gate_cuda(
        pair_output.contiguous().to(torch.float16),
        weights.contiguous().to(device=pair_output.device, dtype=torch.float32),
        shared.contiguous().to(device=pair_output.device, dtype=torch.float16),
        gate_logits.reshape(-1, 1).contiguous().to(
            device=pair_output.device, dtype=torch.float32
        ),
    )


def moe_ffn(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    gate: ExpertWiseWeight,
    up: ExpertWiseWeight,
    down: ExpertWiseWeight,
    *,
    top_k_count: int,
    use_sigmoid: bool = False,
    use_sqrt_softplus: bool = False,
    normalize: bool = False,
    delayed_softmax: bool = False,
    router_bias: torch.Tensor | None = None,
    router_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute a complete routed SwiGLU FFN with expert-wise precision."""

    if gate.n_experts != up.n_experts or gate.n_experts != down.n_experts:
        raise ValueError("MoE FFN expert counts differ")
    if gate.out_per_expert != up.out_per_expert:
        raise ValueError("MoE gate/up widths differ")
    if down.neuron_len != gate.out_per_expert:
        raise ValueError("MoE down input width does not match gate/up output width")
    ids, weights = topk(
        router_logits,
        top_k_count,
        use_sigmoid=use_sigmoid,
        use_sqrt_softplus=use_sqrt_softplus,
        normalize=normalize,
        delayed_softmax=delayed_softmax,
        bias=router_bias,
        scale=router_scale,
    )
    route = MoeRoutePlan.build(ids, gate.n_experts)
    gate_pair = grouped_matmul(gate, x, route)
    up_pair = grouped_matmul(up, x, route)
    hidden = silu_mul(gate_pair, up_pair)
    down_pair = grouped_matmul(down, hidden, route)
    return weighted_reduce(down_pair, weights), ids, weights


def pools_from_groups(groups: Iterable[tuple[dict, Iterable[int]]]) -> tuple[NintExpertPool, ...]:
    """Build pools while preserving each cohort's local expert row order."""

    return tuple(NintExpertPool(weight, tuple(int(v) for v in ids)) for weight, ids in groups)


def to_gpu(
    tensor, device: str | torch.device = "cuda"
) -> ExpertWiseNintWeight | ExpertWiseMixedWeight:
    """Upload an :class:`NintMoeTensor` as native execution cohorts."""

    from mfq.formats.nepq import NepqTensor
    from mfq.formats.moe import NintMoeTensor
    from mfq.quantize.nint_quant import NintTensor
    from mfq.kernels.cuda.nepq_matmul import to_gpu_nepq
    from mfq.kernels.cuda.nvq_matmul import to_gpu_nvq
    from mfq.kernels.torch_backend import to_gpu as nint_to_gpu

    if not isinstance(tensor, NintMoeTensor):
        raise TypeError("to_gpu expects NintMoeTensor")
    if all(isinstance(pool.tensor, NintTensor) for pool in tensor.pools):
        pools = tuple(
            NintExpertPool(
                nint_to_gpu(pool.tensor, device=device, layout="deploy"),
                tuple(int(value) for value in pool.expert_ids),
            )
            for pool in tensor.pools
        )
        return ExpertWiseNintWeight(
            n_experts=tensor.n_experts,
            out_per_expert=tensor.out_per_expert,
            neuron_len=tensor.neuron_len,
            pools=pools,
        )

    mixed_pools: list[CompactExpertPool] = []
    for pool in tensor.pools:
        expert_ids = tuple(int(value) for value in pool.expert_ids)
        if isinstance(pool.tensor, NintTensor):
            family = "nint"
            packed = nint_to_gpu(pool.tensor, device=device, layout="deploy")
        elif isinstance(pool.tensor, NepqTensor):
            family = "nepq"
            packed = to_gpu_nepq(pool.tensor, device=device)
        else:
            family = "nvq"
            packed = to_gpu_nvq(pool.tensor, device=device)
        mixed_pools.append(
            CompactExpertPool(
                family=family,
                weight=packed,
                expert_ids=expert_ids,
            )
        )
    return ExpertWiseMixedWeight(
        n_experts=tensor.n_experts,
        out_per_expert=tensor.out_per_expert,
        neuron_len=tensor.neuron_len,
        pools=tuple(mixed_pools),
    )
