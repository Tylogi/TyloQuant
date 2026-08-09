"""TPQ grouped GEMM: combine matrix multiplies for a layer's selected top-k MoE experts into a few batched operators.

Background: all top-6 experts in every DSV4-S layer are VQWeight objects with u8 indices and layer-shared codebooks.
The original implementation called VQWeight.matmul_T in a per-expert loop. Decoding one layer used six experts times
two gu/dn matmuls plus gather and reduction, or about 30+ kernel launches. Across 43 layers this reached thousands,
making launch overhead a primary bottleneck under WDDM.

This module batches all selected experts in one layer and computes:
  s = xb @ cb^T          -- batched codebook dot products [N, B, K]
  g = gather(s, idx^T)   -- batched lookup               [N, B, R]
  y = g.sum(1)           -- batched reduction            [N, R]
This reduces each layer from 30+ launches to about eight. Numerics match the VQWeight.matmul_T LUT algorithm
element by element using the same f32 expression, only batched. cuBLAS reduction-order differences are about 1e-6,
far below VQ quantization noise.

Interfaces:
  vq_gemv_batch    -- generic batched VQ matrix multiplication for T=1 or pairwise T>1 mode
  stack_vq         -- stack several (gu, dn) experts into batched tensors
  moe_mlp_grouped  -- complete SwiGLU MLP for one layer's top-k experts, including routing-weighted reduction
Prefer fused CUDA kernels from fusedext when compiled; otherwise use the torch batched path in this file.

Self-check, runnable on CPU: python -m tpq.grouped
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

from .kernels import VQWeight, cb_compute
from .precision import compute_dtype

_slots_fused = None
_slots_fused_checked = False


def _load_slots_fused():
    """Load the CUDA extension only for an actual CUDA MoE call.

    The shared grouped module also provides CPU SiTU/packed-MoE helpers. Compiling CUDA during module import
    would make CPU-only CLI startup and unit tests depend unnecessarily on NVCC. Cache one probe result here;
    CUDA behavior remains unchanged while the CPU path never touches fusedext.
    """
    global _slots_fused, _slots_fused_checked
    if _slots_fused_checked:
        return _slots_fused
    _slots_fused_checked = True
    try:
        from . import fusedext as extension

        if extension.available():
            _slots_fused = extension.moe_mlp_slots_fused
    except Exception:
        _slots_fused = None
    return _slots_fused


_SLOT_WORKSPACES: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_CPU_PACKED_PROFILE = {
    "gu_seconds": 0.0,
    "activation_seconds": 0.0,
    "down_seconds": 0.0,
    "reduce_seconds": 0.0,
    "groups": 0,
    "fused_seconds": 0.0,
    "fused_calls": 0,
}


def reset_cpu_packed_profile() -> None:
    for key in _CPU_PACKED_PROFILE:
        _CPU_PACKED_PROFILE[key] = (
            0 if key in {"groups", "fused_calls"} else 0.0
        )
    try:
        from .cpuext import reset_packed_moe_phase_profile

        reset_packed_moe_phase_profile()
    except (ImportError, RuntimeError):
        pass


def cpu_packed_profile() -> dict[str, float | int]:
    result = dict(_CPU_PACKED_PROFILE)
    try:
        from .cpuext import packed_moe_phase_profile

        result.update(
            {
                f"fused_{key}": value
                for key, value in packed_moe_phase_profile().items()
            }
        )
    except (ImportError, RuntimeError):
        pass
    return result


def activate_gate_up(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    activation: str = "silu",
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
) -> torch.Tensor:
    """Apply the model-specific gated activation without dequantizing weights."""
    if gate.dtype == torch.bfloat16:
        from .ops import gated_activation

        fused = gated_activation(
            gate,
            up,
            activation=activation,
            beta=situ_beta,
            linear_beta=situ_linear_beta,
        )
        if fused is not None:
            return fused
    if activation in {"silu", "swiglu"}:
        return F.silu(gate) * up
    if activation != "situ":
        raise ValueError(f"unsupported expert activation {activation!r}")
    gate_f = gate.float()
    up_f = up.float()
    activated = (
        float(situ_beta)
        * torch.tanh(gate_f / float(situ_beta))
        * torch.sigmoid(gate_f)
    )
    if situ_linear_beta is not None:
        linear_beta = float(situ_linear_beta)
        up_f = linear_beta * torch.tanh(up_f / linear_beta)
    return (activated * up_f).to(gate.dtype)


def moe_mlp_slots_compatible(
    experts: list[tuple[VQWeight, VQWeight]],
) -> bool:
    """Whether the SM120 slot kernel can process all experts in one call."""
    if not experts or len(experts) > 8:
        return False
    first_gu, first_dn = experts[0]
    return not any(
        gu.idx.shape[0] != first_gu.idx.shape[0]
        or gu.cols != first_gu.cols
        or gu.idx.dtype != first_gu.idx.dtype
        or dn.idx.shape[0] != first_dn.idx.shape[0]
        or dn.cols != first_dn.cols
        or dn.idx.dtype != first_dn.idx.dtype
        for gu, dn in experts
    )


def vq_gemv_batch(x_rows: torch.Tensor, idx: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
    """Batched VQ matrix multiplication y[n] = x[n] @ W_n^T, with W_n defined by idx[n]/cb[n].

    x_rows: f32 [N, C] or [1, C] broadcast to N; idx: u8/u16 [N, R, B].
    cb: f32 [N, K, dim] or [1, K, dim], broadcasting a layer-shared codebook without N stacked copies.
    Returns f32 [N, R]. C = B * dim.
    """
    N, R, B = (max(x_rows.shape[0], idx.shape[0]),) + idx.shape[1:]
    dim = cb.shape[2]
    if idx.dtype in (torch.uint8, torch.uint16):
        # CPU and CUDA both use the same capability registry; use the torch reference path only when the backend is unavailable.
        from .ops import vq_gemv

        registered = vq_gemv(x_rows, idx, cb)
        if registered is not None:
            return registered
    dt = compute_dtype(x_rows.device)
    cbc = cb_compute(cb, dt)
    xb = x_rows.to(dt).view(-1, B, dim)
    s = torch.matmul(xb, cbc.transpose(1, 2))          # [N|1, B, K] (broadcast; half-precision GEMM)
    if s.shape[0] == 1 and N > 1:
        s = s.expand(N, -1, -1)                        # Explicitly expand to N when both x and cb are broadcast
    gi = idx.long().permute(0, 2, 1)
    if gi.shape[0] == 1 and N > 1:
        gi = gi.expand(N, -1, -1)                      # Broadcast idx (x[N] x idx[1] mode)
    g = s.gather(2, gi)                                # [N, B, R]
    return g.sum(1, dtype=torch.float32)               # Accumulate in f32


def _stack_cb(cbs: list[torch.Tensor]) -> torch.Tensor:
    """Stack codebooks. When all experts share one tensor in the same layer/tier, return a [1,K,d] broadcast view
    to avoid top-k D2D codebook copies per layer; perform a real stack only for mixed codebooks."""
    first = cbs[0]
    if all(c is first or (c.data_ptr() == first.data_ptr() and c.shape == first.shape)
           for c in cbs):
        return first.unsqueeze(0)
    return torch.stack(cbs)


def stack_vq(experts: list[tuple[VQWeight, VQWeight]]):
    """Stack top-k (gu, dn) experts into batched tensors. u8 indices use about 17 MB of D2D copies per layer,
    which is negligible; shared codebooks use one broadcast view.

    Returns (gu_idx [N,Rg,Bg], gu_cb [N|1,K,d], dn_idx [N,Rd,Bd], dn_cb [N|1,K,d]).
    """
    gu_idx = torch.stack([g.idx for g, _ in experts])
    gu_cb = _stack_cb([g.cb for g, _ in experts])
    dn_idx = torch.stack([d.idx for _, d in experts])
    dn_cb = _stack_cb([d.cb for _, d in experts])
    return gu_idx, gu_cb, dn_idx, dn_cb


def _slot_workspace(
    device: torch.device,
    count: int,
    gu_rows: int,
    dn_rows: int,
    signature_tag: tuple,
    result_dtype: torch.dtype = torch.bfloat16,
):
    key = (
        str(device),
        count,
        gu_rows,
        dn_rows,
        signature_tag,
        result_dtype,
    )
    workspace = _SLOT_WORKSPACES.get(key)
    if workspace is None:
        workspace = (
            torch.empty(
                count, gu_rows, device=device, dtype=torch.bfloat16
            ),
            torch.empty(
                count, dn_rows, device=device, dtype=torch.bfloat16
            ),
            torch.empty(
                dn_rows, device=device, dtype=result_dtype
            ),
        )
        _SLOT_WORKSPACES[key] = workspace
    return workspace


def moe_mlp_grouped_slots(
    x_rows: torch.Tensor,
    experts: list[tuple[VQWeight, VQWeight]],
    weights: torch.Tensor,
    limit: float = 0.0,
    result_dtype: torch.dtype = torch.bfloat16,
    activation: str = "silu",
) -> torch.Tensor | None:
    """SM120 top-k VQ MLP that directly reads fixed-arena views.

    GU/DN indices are no longer copied with ``torch.stack``. Four launches perform GU VQ GEMV, SwiGLU,
    DN VQ GEMV, and FP32 routing weighting, with BF16 output between layers. Return None for mixed signatures,
    non-CUDA execution, or an unavailable extension so the caller uses the original batched implementation.
    """
    slots_fused = _load_slots_fused() if x_rows.is_cuda else None
    if (
        slots_fused is None
        or os.environ.get("TPQ_SLOT_VQ", "1") == "0"
        or not x_rows.is_cuda
        or not experts
        or len(experts) > 8
        or x_rows.shape[0] != 1
        or activation not in {"silu", "swiglu"}
    ):
        return None
    # v/w archives may use different code dimensions (4 vs 8) for the same
    # logical matrix.  The SM120 kernel accepts per-expert block/codebook
    # dimensions, so only logical row/column sizes and index dtypes must agree.
    if not moe_mlp_slots_compatible(experts):
        return None
    first_gu, first_dn = experts[0]
    gu_codebooks = [
        cb_compute(gu.cb, torch.bfloat16).contiguous()
        for gu, _ in experts
    ]
    dn_codebooks = [
        cb_compute(dn.cb, torch.bfloat16).contiguous()
        for _, dn in experts
    ]
    x_bf16 = x_rows.to(torch.bfloat16)
    route_weights = weights.float().contiguous()
    hidden_workspace, out_workspace, result = _slot_workspace(
        x_rows.device,
        len(experts),
        first_gu.idx.shape[0],
        first_dn.idx.shape[0],
        (
            tuple(
                (
                    tuple(gu.idx.shape),
                    tuple(gu.cb.shape),
                    tuple(dn.idx.shape),
                    tuple(dn.cb.shape),
                )
                for gu, dn in experts
            ),
            str(first_gu.idx.dtype),
            str(first_dn.idx.dtype),
        ),
        result_dtype,
    )
    return slots_fused(
        x_bf16,
        [gu.idx for gu, _ in experts],
        gu_codebooks,
        [dn.idx for _, dn in experts],
        dn_codebooks,
        route_weights,
        limit,
        hidden_workspace,
        out_workspace,
        result,
    )


def moe_mlp_grouped_partial(
    x_rows: torch.Tensor,
    experts: list[tuple[VQWeight, VQWeight]],
    weights: torch.Tensor,
    limit: float = 0.0,
    *,
    activation: str = "silu",
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
) -> torch.Tensor:
    """Return one FP32 route-weighted hidden partial for a local EP rank."""
    groups: dict[tuple, list[int]] = {}
    for i, (gu, dn) in enumerate(experts):
        key = (
            gu.idx.shape,
            gu.cb.shape,
            gu.idx.dtype,
            dn.idx.shape,
            dn.cb.shape,
            dn.idx.dtype,
        )
        groups.setdefault(key, []).append(i)
    result = None
    for positions in groups.values():
        selected = [experts[position] for position in positions]
        index = torch.tensor(positions, device=weights.device)
        local_weights = weights[index].contiguous()
        part = moe_mlp_grouped_slots(
            x_rows,
            selected,
            local_weights,
            limit,
            result_dtype=torch.float32,
            activation=activation,
        )
        if part is None:
            part = moe_mlp_grouped(
                x_rows.float(),
                *stack_vq(selected),
                local_weights,
                limit,
                activation=activation,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
            )
        if result is None:
            result = part.float()
        else:
            result.add_(part.float())
    if result is None:
        raise ValueError("at least one expert is required")
    return result


def moe_mlp_grouped_mixed(
    x_rows: torch.Tensor,
    experts: list[tuple[VQWeight, ...]],
    weights: torch.Tensor,
    limit: float = 0.0,
    *,
    activation: str = "silu",
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
) -> torch.Tensor:
    """Mixed v/w-tier compatible path for DSV4-S per-layer mixtures. v=4D and w=8D have different B and cannot stack,
    so group by gu/dn shapes, batch separately, and sum by routing weights. One group is equivalent to moe_mlp_grouped.

    x_rows: [N, D] or [1, D]; experts: top-k (gu, dn) pairs; weights: [K].
    Returns an f32 weighted sum [D].
    """
    if not experts:
        raise ValueError("at least one expert is required")
    arity = len(experts[0])
    if arity not in (2, 3) or any(
        len(bundle) != arity for bundle in experts
    ):
        raise ValueError("expert projections must consistently be 2 or 3")
    slotted = (
        moe_mlp_grouped_slots(
            x_rows,
            experts,
            weights,
            limit,
            activation=activation,
        )
        if arity == 2
        else None
    )
    if slotted is not None:
        return slotted
    packed = hasattr(experts[0][0], "raw")
    if any(
        hasattr(weight, "raw") != packed
        for bundle in experts
        for weight in bundle
    ):
        raise ValueError("packed and expanded VQ experts cannot be mixed")
    if (
        packed
        and not x_rows.is_cuda
        and os.environ.get("TPQ_CPU_PACKED_MOE", "1") != "0"
    ):
        from .ops import packed_moe_selected_topk

        profile = os.environ.get("TPQ_CPU_PACKED_PROFILE", "0") != "0"
        started = time.perf_counter() if profile else 0.0
        fused = packed_moe_selected_topk(
            x_rows,
            experts,
            weights,
            activation=activation,
            activation_beta=situ_beta,
            activation_linear_beta=situ_linear_beta,
            limit=limit,
        )
        if fused is not None:
            if profile:
                _CPU_PACKED_PROFILE["fused_seconds"] += (
                    time.perf_counter() - started
                )
                _CPU_PACKED_PROFILE["fused_calls"] += 1
            return fused
    if packed and arity == 3 and not x_rows.is_cuda:
        from .ops import vq_gemv_packed_list

        gate = experts[0][0]
        up = experts[0][1]
        down = experts[0][2]
        same_layout = all(
            (
                current.rows,
                current.blocks,
                current.bits,
                tuple(current.cb.shape),
                current.cb.data_ptr(),
            )
            == (
                reference.rows,
                reference.blocks,
                reference.bits,
                tuple(reference.cb.shape),
                reference.cb.data_ptr(),
            )
            for bundle in experts
            for current, reference in zip(
                bundle,
                (gate, up, down),
            )
        )
        if same_layout:
            gate_values = vq_gemv_packed_list(
                x_rows.float(),
                [bundle[0].raw for bundle in experts],
                gate.cb,
                gate.rows,
                gate.blocks,
                gate.bits,
            )
            up_values = vq_gemv_packed_list(
                x_rows.float(),
                [bundle[1].raw for bundle in experts],
                up.cb,
                up.rows,
                up.blocks,
                up.bits,
            )
            if gate_values is not None and up_values is not None:
                if limit:
                    gate_values.clamp_(max=limit)
                    up_values.clamp_(-limit, limit)
                activated = activate_gate_up(
                    gate_values,
                    up_values,
                    activation=activation,
                    situ_beta=situ_beta,
                    situ_linear_beta=situ_linear_beta,
                )
                part = vq_gemv_packed_list(
                    activated,
                    [bundle[2].raw for bundle in experts],
                    down.cb,
                    down.rows,
                    down.blocks,
                    down.bits,
                    allow_direct=True,
                )
                if part is not None:
                    return (
                        part
                        * weights.float().reshape(-1, 1)
                    ).sum(0)

        # Compiler-less correctness path. Expanded indices and matrices are
        # temporary and never enter the model cache.
        result = torch.zeros(
            x_rows.shape[-1],
            dtype=torch.float32,
            device=x_rows.device,
        )
        source = x_rows[0].float()
        for expert, bundle in enumerate(experts):
            gate_weight, up_weight, down_weight = bundle

            def apply(weight, value):
                indices = weight.unpack().long()
                matrix = (
                    weight.cb[indices]
                    .reshape(weight.rows, weight.cols)
                    .float()
                )
                return torch.mv(matrix, value)

            gate_value = apply(gate_weight, source)
            up_value = apply(up_weight, source)
            if limit:
                gate_value.clamp_(max=limit)
                up_value.clamp_(-limit, limit)
            activated = activate_gate_up(
                gate_value,
                up_value,
                activation=activation,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
            )
            result.add_(
                apply(down_weight, activated.float()),
                alpha=float(weights[expert]),
            )
        return result
    if arity != 2:
        raise ValueError(
            "three-projection experts require packed CPU execution"
        )
    groups: dict[tuple, list[int]] = {}
    for i, (g, d) in enumerate(experts):
        if packed:
            key = (
                g.rows,
                g.blocks,
                g.bits,
                g.cb.shape,
                g.cb.data_ptr(),
                d.rows,
                d.blocks,
                d.bits,
                d.cb.shape,
                d.cb.data_ptr(),
            )
        else:
            key = (
                g.idx.shape,
                g.cb.shape,
                g.idx.dtype,
                d.idx.shape,
                d.cb.shape,
                d.idx.dtype,
            )
        groups.setdefault(key, []).append(i)
    y = None
    for idxs in groups.values():
        elist = [experts[i] for i in idxs]
        w = weights[torch.tensor(idxs, device=weights.device)]
        if packed and not x_rows.is_cuda:
            from .ops import vq_gemv_packed_list

            gu = elist[0][0]
            dn = elist[0][1]
            profile = os.environ.get("TPQ_CPU_PACKED_PROFILE", "0") != "0"
            started = time.perf_counter() if profile else 0.0
            h = vq_gemv_packed_list(
                x_rows.float(),
                [weight.raw for weight, _ in elist],
                gu.cb,
                gu.rows,
                gu.blocks,
                gu.bits,
            )
            if h is not None:
                activated_at = time.perf_counter() if profile else 0.0
                gate, up = h[:, :dn.cols], h[:, dn.cols:]
                if limit:
                    up = up.clamp(-limit, limit)
                    gate = gate.clamp(max=limit)
                activated = activate_gate_up(
                    gate,
                    up,
                    activation=activation,
                    situ_beta=situ_beta,
                    situ_linear_beta=situ_linear_beta,
                )
                down_at = time.perf_counter() if profile else 0.0
                part = vq_gemv_packed_list(
                    activated,
                    [weight.raw for _, weight in elist],
                    dn.cb,
                    dn.rows,
                    dn.blocks,
                    dn.bits,
                    allow_direct=(
                        os.environ.get(
                            "TPQ_CPU_PACKED_DIRECT",
                            "1",
                        )
                        != "0"
                    ),
                )
                if part is not None:
                    reduce_at = time.perf_counter() if profile else 0.0
                    part = (part * w.float().unsqueeze(1)).sum(0)
                    if profile:
                        finished = time.perf_counter()
                        _CPU_PACKED_PROFILE["gu_seconds"] += (
                            activated_at - started
                        )
                        _CPU_PACKED_PROFILE["activation_seconds"] += (
                            down_at - activated_at
                        )
                        _CPU_PACKED_PROFILE["down_seconds"] += (
                            reduce_at - down_at
                        )
                        _CPU_PACKED_PROFILE["reduce_seconds"] += (
                            finished - reduce_at
                        )
                        _CPU_PACKED_PROFILE["groups"] += 1
                    y = part if y is None else y + part.to(y.dtype)
                    continue
            # Compiler-less CPU environments retain a correctness fallback,
            # but the expanded tensors are temporary and never enter the pool.
            from .kernels import VQWeight

            unpacked = [
                (
                    VQWeight(gu.unpack(), gu.cb, gu.cols),
                    VQWeight(dn.unpack(), dn.cb, dn.cols),
                )
                for gu, dn in elist
            ]
            part = moe_mlp_grouped_mixed(
                x_rows,
                unpacked,
                w,
                limit,
                activation=activation,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
            )
            y = part if y is None else y + part.to(y.dtype)
            continue
        if not x_rows.is_cuda:
            from .cpuext import vq_gemv_list_cpu

            gu_cb = elist[0][0].cb
            dn_cb = elist[0][1].cb
            shared_gu = all(
                gu.cb.data_ptr() == gu_cb.data_ptr() for gu, _ in elist
            )
            shared_dn = all(
                dn.cb.data_ptr() == dn_cb.data_ptr() for _, dn in elist
            )
            if shared_gu and shared_dn:
                h = vq_gemv_list_cpu(
                    x_rows.float(), [gu.idx for gu, _ in elist], gu_cb
                )
                if h is not None:
                    mi = elist[0][1].cols
                    gate, up = h[:, :mi], h[:, mi:]
                    if limit:
                        up = up.clamp(-limit, limit)
                        gate = gate.clamp(max=limit)
                    activated = activate_gate_up(
                        gate,
                        up,
                        activation=activation,
                        situ_beta=situ_beta,
                        situ_linear_beta=situ_linear_beta,
                    )
                    part = vq_gemv_list_cpu(
                        activated,
                        [dn.idx for _, dn in elist],
                        dn_cb,
                    )
                    if part is not None:
                        part = (part * w.float().unsqueeze(1)).sum(0)
                        y = part if y is None else y + part.to(y.dtype)
                        continue
        part = moe_mlp_grouped_slots(
            x_rows,
            elist,
            w,
            limit,
            activation=activation,
        )
        if part is None:
            part = moe_mlp_grouped(
                x_rows.float(),
                *stack_vq(elist),
                w,
                limit,
                activation=activation,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
            )
        if y is None:
            y = part
        else:
            y += part.to(y.dtype)
    return y


def moe_mlp_grouped_situ(
    x_rows: torch.Tensor,
    experts: list[tuple[VQWeight, VQWeight]],
    weights: torch.Tensor,
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
) -> torch.Tensor:
    """Stable direct entry point for Kimi/TPQ: mixed-tier VQ MoE plus SiTU activation.

    Retain a simple positional signature so quantizer runtime need not understand TPQ's generic
    ``activation``/``limit`` options and will not regress when shared grouped operators are refactored.
    """
    return moe_mlp_grouped_mixed(
        x_rows,
        experts,
        weights,
        activation="situ",
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )


def expert_mlp_batched(
    x_rows: torch.Tensor,
    gu: VQWeight,
    dn: VQWeight,
    limit: float = 0.0,
    *,
    activation: str = "silu",
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
) -> torch.Tensor:
    """Multi-token by single-expert SwiGLU MLP for the speculative-validation T>1 path.

    Broadcast idx as [1,R,B] with idxStrideN=0 in the fused kernel and compute x_rows [Tn, D] in one pass.
    Numerics match per-token VQWeight.matmul_T calls. Returns f32 [Tn, D_out].
    """
    mi = dn.cols
    h = vq_gemv_batch(x_rows, gu.idx.unsqueeze(0), gu.cb.unsqueeze(0))   # [Tn, 2*mi]
    g, u = h[:, :mi], h[:, mi:]
    if limit:
        u = u.clamp(-limit, limit)
        g = g.clamp(max=limit)
    activated = activate_gate_up(
        g,
        u,
        activation=activation,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    return vq_gemv_batch(
        activated,
        dn.idx.unsqueeze(0),
        dn.cb.unsqueeze(0),
    )


def moe_mlp_grouped(
    x_rows: torch.Tensor,
    gu_idx: torch.Tensor,
    gu_cb: torch.Tensor,
    dn_idx: torch.Tensor,
    dn_cb: torch.Tensor,
    weights: torch.Tensor,
    limit: float = 0.0,
    *,
    activation: str = "silu",
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
) -> torch.Tensor:
    """SwiGLU MLP for one layer's top-k experts, numerically matching the per-expert loop in DSV4TPQModel._expert_mlp_tpq.

    x_rows: [N, D] or [1, D], where N is the number of expert pairs; weights: [N] routing weights.
    Returns an f32 weighted sum [D].
    """
    mi = dn_idx.shape[2] * dn_cb.shape[2]              # Number of dn columns = moe_inter
    h = vq_gemv_batch(x_rows, gu_idx, gu_cb)           # [N, 2*mi]
    g, u = h[:, :mi], h[:, mi:]
    if limit:
        u = u.clamp(-limit, limit)
        g = g.clamp(max=limit)
    a = activate_gate_up(
        g,
        u,
        activation=activation,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    out = vq_gemv_batch(a, dn_idx, dn_cb)              # [N, D]
    return (out * weights.unsqueeze(1)).sum(0)         # [D]


def _selftest() -> None:
    """Compare numerically against a per-expert VQWeight.matmul_T loop using random CPU data."""
    torch.manual_seed(7)
    N, D, mi, K, dim, limit = 6, 256, 96, 16, 8, 7.0

    def mk(rows: int, cols: int) -> VQWeight:
        idx = torch.randint(0, K, (rows, cols // dim), dtype=torch.uint8)
        cb = torch.randn(K, dim) * 0.05
        return VQWeight(idx, cb, cols)

    x = torch.randn(1, D)
    wts = torch.rand(N)
    experts = [(mk(2 * mi, D), mk(D, mi)) for _ in range(N)]

    # Reference: per-expert loop (the original dsv4model._moe path)
    ref = torch.zeros(D)
    for (gu, dn), w in zip(experts, wts):
        h = gu.matmul_T(x)
        g, u = h[:, :mi], h[:, mi:]
        u = u.clamp(-limit, limit)
        g = g.clamp(max=limit)
        ref += dn.matmul_T(F.silu(g) * u).squeeze(0) * w

    got = moe_mlp_grouped(x, *stack_vq(experts), wts, limit)
    diff = (ref - got).abs().max().item()
    rel = diff / ref.abs().max().item()
    print(f"grouped 自检: 最大绝对误差 {diff:.3e}（相对 {rel:.3e}）")
    assert rel < 1e-5, f"分组 GEMM 与逐专家循环不一致: rel={rel}"

    # Mixed tiers (v=4D / w=8D, the actual DSV4-S per-layer form)
    dim2 = 4
    def mk2(rows: int, cols: int) -> VQWeight:
        idx = torch.randint(0, K, (rows, cols // dim2), dtype=torch.uint8)
        cb = torch.randn(K, dim2) * 0.05
        return VQWeight(idx, cb, cols)
    experts_mx = []
    for i in range(N):
        gu = mk(2 * mi, D) if i % 2 else mk2(2 * mi, D)
        dn = mk(D, mi) if i % 2 else mk2(D, mi)
        experts_mx.append((gu, dn))
    ref_mx = torch.zeros(D)
    for (gu, dn), w in zip(experts_mx, wts):
        h = gu.matmul_T(x)
        g, u = h[:, :mi], h[:, mi:]
        u = u.clamp(-limit, limit)
        g = g.clamp(max=limit)
        ref_mx += dn.matmul_T(F.silu(g) * u).squeeze(0) * w
    got_mx = moe_mlp_grouped_mixed(x, experts_mx, wts, limit)
    rel_mx = (ref_mx - got_mx).abs().max().item() / ref_mx.abs().max().item()
    print(f"混档自检: 相对误差 {rel_mx:.3e}")
    assert rel_mx < 1e-5, f"混档分组 GEMM 不一致: rel={rel_mx}"
    print("grouped 自检通过")


if __name__ == "__main__":
    _selftest()
