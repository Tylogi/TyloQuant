"""Fixed-pool GPU quantizer for the four NEPQ profiles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from mfq.formats.nepq import (
    NEPQ0_L,
    NEPQ0_S,
    NEPQ1_L,
    NEPQ1_S,
    NepqSpec,
    NepqTensor,
    nepq_spec,
    rotation_signs,
    validate_nepq,
)
from mfq.formats.npq0_l import unpack_npq0_l_tables
from mfq.formats.npq0_s import unpack_npq0_s_tables
from mfq.formats.nvq1_l import unpack_ternary_codebook
from mfq.formats.nvq1_s import unpack_nvq1_s_banked_codebook


@dataclass(frozen=True)
class NepqQuantConfig:
    anchor_multipliers: tuple[float, ...] = (0.75, 1.0, 1.25)
    refine_steps: int = 2
    row_chunk: int = 8
    bank_chunk: int = 8
    admm_iterations: int = 0
    admm_rho: float = 1.0

    def __post_init__(self) -> None:
        if not self.anchor_multipliers or any(
            not np.isfinite(value) or value <= 0
            for value in self.anchor_multipliers
        ):
            raise ValueError("NEPQ anchor multipliers must be finite and positive")
        if self.refine_steps < 0:
            raise ValueError("NEPQ refine_steps must be non-negative")
        if self.row_chunk <= 0 or self.bank_chunk <= 0:
            raise ValueError("NEPQ row_chunk and bank_chunk must be positive")
        if self.admm_iterations < 0:
            raise ValueError("NEPQ ADMM iterations must be non-negative")
        if not np.isfinite(self.admm_rho) or self.admm_rho <= 0:
            raise ValueError("NEPQ ADMM rho must be finite and positive")


@dataclass
class _Pool:
    spec: NepqSpec
    codes: torch.Tensor | None
    scale: torch.Tensor | None
    maximum_basis: float
    first_codes: torch.Tensor | None = None
    second_codes: torch.Tensor | None = None
    native_scale: torch.Tensor | None = None
    native_first_codes: torch.Tensor | None = None
    native_second_codes: torch.Tensor | None = None

    @property
    def banks(self) -> int:
        source = self.codes if self.codes is not None else self.first_codes
        assert source is not None
        return int(source.shape[0])


@dataclass
class _Assignment:
    bank_ids: torch.Tensor
    state: torch.Tensor
    indices: torch.Tensor
    aux: torch.Tensor | None
    anchor: torch.Tensor
    row_error: torch.Tensor


def _table_array(spec: NepqSpec, table_payloads) -> np.ndarray:
    if isinstance(table_payloads, np.ndarray):
        result = np.ascontiguousarray(table_payloads, dtype=np.uint8)
    else:
        rows = [np.frombuffer(bytes(value), dtype=np.uint8) for value in table_payloads]
        result = np.stack(rows) if rows else np.empty((0, spec.table_bytes), dtype=np.uint8)
    if result.ndim != 2 or result.shape[1] != spec.table_bytes:
        raise ValueError(
            f"{spec.label} table pool must have shape [banks,{spec.table_bytes}]"
        )
    if not 1 <= result.shape[0] <= 256:
        raise ValueError("NEPQ table pool must contain 1 to 256 banks")
    return result


def _decode_pool(
    spec: NepqSpec,
    table_payloads: np.ndarray,
    device: str | torch.device,
) -> _Pool:
    payloads = [row.tobytes() for row in table_payloads]
    banks = len(payloads)
    if spec is NEPQ0_S:
        unpacked = [unpack_npq0_s_tables(payload)[:3] for payload in payloads]
        scale, first, second = (np.stack(values) for values in zip(*unpacked, strict=True))
        maximum = max(
            float(np.abs(first).max()), float(np.abs(second).max())
        ) * float(scale.max())
        return _Pool(
            spec,
            None,
            torch.as_tensor(scale, device=device, dtype=torch.float32),
            max(maximum, 1e-12),
            torch.as_tensor(first, device=device, dtype=torch.float32),
            torch.as_tensor(second, device=device, dtype=torch.float32),
            torch.as_tensor(
                np.ascontiguousarray(scale.T),
                device=device,
                dtype=torch.float32,
            ),
            torch.as_tensor(
                np.ascontiguousarray(first.transpose(1, 2, 3, 0)),
                device=device,
                dtype=torch.int8,
            ),
            torch.as_tensor(
                np.ascontiguousarray(second.transpose(1, 2, 3, 0)),
                device=device,
                dtype=torch.int8,
            ),
        )
    if spec is NEPQ0_L:
        unpacked = [unpack_npq0_l_tables(payload)[:3] for payload in payloads]
        scale, first, second = (np.stack(values) for values in zip(*unpacked, strict=True))
        maximum = max(
            float(np.abs(first).max()), float(np.abs(second).max())
        ) * float(scale.max())
        return _Pool(
            spec,
            None,
            torch.as_tensor(scale, device=device, dtype=torch.float32),
            max(maximum, 1e-12),
            torch.as_tensor(first, device=device, dtype=torch.float32),
            torch.as_tensor(second, device=device, dtype=torch.float32),
        )
    if spec is NEPQ1_S:
        codes = np.stack([unpack_nvq1_s_banked_codebook(payload) for payload in payloads])
        maximum = 15.0 * (float(np.abs(codes).max()) + 0.15625)
        return _Pool(
            spec,
            torch.as_tensor(codes, device=device, dtype=torch.float32),
            None,
            max(maximum, 1e-12),
        )
    if spec is NEPQ1_L:
        codes = np.stack([unpack_ternary_codebook(payload) for payload in payloads])
        maximum = 7.0 * (float(np.abs(codes).max()) + 0.125)
        return _Pool(
            spec,
            torch.as_tensor(codes, device=device, dtype=torch.float32),
            None,
            max(maximum, 1e-12),
        )
    raise ValueError(f"unsupported NEPQ profile: {spec.label}")


def _fwht_blocks(value: torch.Tensor, block: int) -> torch.Tensor:
    result = value.contiguous().reshape(-1, block).clone()
    stride = 1
    while stride < block:
        paired = result.reshape(-1, 2, stride)
        first = paired[:, 0].clone()
        second = paired[:, 1].clone()
        paired[:, 0] = first + second
        paired[:, 1] = first - second
        stride *= 2
    result.mul_(1.0 / math.sqrt(block))
    return result.reshape_as(value)


def _source_shape(weight) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in weight.shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("NEPQ weight must have shape [experts,out,K]")
    if shape[2] % 8:
        raise ValueError("NEPQ K must be divisible by 8")
    return shape


def _slice_rows(source, start: int, stop: int, width: int, device) -> torch.Tensor:
    if hasattr(source, "read_rows"):
        value = source.read_rows(start, stop, device=device)
        if tuple(int(item) for item in value.shape) != (stop - start, width):
            raise ValueError(
                f"NEPQ row source returned {tuple(value.shape)}, "
                f"expected {(stop - start, width)}"
            )
        return value.to(device=device, dtype=torch.float32).contiguous()
    flat = source.reshape(-1, width)
    if isinstance(flat, torch.Tensor):
        return flat[start:stop].to(device=device, dtype=torch.float32).contiguous()
    value = np.asarray(flat[start:stop], dtype=np.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32).contiguous()


def _importance_slice(
    importance,
    start: int,
    stop: int,
    shape: tuple[int, int, int],
    device,
) -> torch.Tensor:
    rows = stop - start
    width = shape[2]
    if importance is None:
        return torch.ones((rows, width), device=device, dtype=torch.float32)
    if tuple(int(value) for value in importance.shape) == (width,):
        if isinstance(importance, torch.Tensor):
            value = importance.to(device=device, dtype=torch.float32)
        else:
            value = torch.as_tensor(importance, device=device, dtype=torch.float32)
        result = value[None].expand(rows, -1).contiguous()
    elif tuple(int(value) for value in importance.shape) == (
        shape[0],
        width,
    ):
        expert_ids = torch.arange(
            start, stop, device=device, dtype=torch.int64
        ).div(shape[1], rounding_mode="floor")
        if isinstance(importance, torch.Tensor):
            value = importance.to(device=device, dtype=torch.float32)
        else:
            value = torch.as_tensor(
                importance, device=device, dtype=torch.float32
            )
        result = value.index_select(0, expert_ids).contiguous()
    elif tuple(int(value) for value in importance.shape) == shape:
        result = _slice_rows(importance, start, stop, width, device)
    else:
        raise ValueError(
            f"NEPQ importance must have shape [{width}], "
            f"[{shape[0]},{width}], or {shape}"
        )
    if not torch.isfinite(result).all() or torch.any(result < 0):
        raise ValueError("NEPQ importance must be finite and non-negative")
    return result


def _score_npq_banks(
    xvec: torch.Tensor,
    wvec: torch.Tensor,
    constant: torch.Tensor,
    anchor: torch.Tensor,
    rows: int,
    ng: int,
    pool: _Pool,
    start: int,
    stop: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    assert pool.first_codes is not None and pool.second_codes is not None
    assert pool.scale is not None
    count = stop - start
    states = 1 << pool.spec.state_bits
    best_error = torch.full(
        (rows, ng, count), torch.inf, device=xvec.device, dtype=torch.float32
    )
    best_state = torch.zeros((rows, ng, count), device=xvec.device, dtype=torch.uint8)
    best_indices = torch.zeros(
        (rows, ng, count, 3), device=xvec.device, dtype=torch.int64
    )
    first_x, second_x = xvec[:, :4], xvec[:, 4:]
    first_w, second_w = wvec[:, :4], wvec[:, 4:]
    weighted_first = first_w * first_x
    weighted_second = second_w * second_x
    first_constant = (first_w * first_x.square()).sum(1)
    second_constant = (second_w * second_x.square()).sum(1)
    group_anchor = anchor.repeat_interleave(ng * 3)
    for state in range(states):
        first_table = pool.first_codes[start:stop, state]
        second_table = pool.second_codes[start:stop, state]
        first_flat = first_table.reshape(-1, 4)
        second_flat = second_table.reshape(-1, 4)
        first_cross = (weighted_first @ first_flat.T).reshape(
            -1, count, first_table.shape[1]
        )
        first_quad = (first_w @ first_flat.square().T).reshape_as(
            first_cross
        )
        second_cross = (weighted_second @ second_flat.T).reshape(
            -1, count, second_table.shape[1]
        )
        second_quad = (second_w @ second_flat.square().T).reshape_as(
            second_cross
        )
        scale = group_anchor[:, None] * pool.scale[start:stop, state][None, :]
        scale_vector = scale[:, :, None]
        first_variable = (
            scale_vector.square() * first_quad
            - 2.0 * scale_vector * first_cross
        )
        second_variable = (
            scale_vector.square() * second_quad
            - 2.0 * scale_vector * second_cross
        )
        first_minimum, first_indices = first_variable.min(2)
        second_minimum, second_indices = second_variable.min(2)
        vector_error = (
            first_constant[:, None]
            + second_constant[:, None]
            + first_minimum
            + second_minimum
        )
        error = vector_error.reshape(rows, ng, 3, count).sum(2)
        improve = error < best_error
        best_error = torch.where(improve, error, best_error)
        best_state = torch.where(
            improve, torch.full_like(best_state, state), best_state
        )
        composite = first_indices | (second_indices << 3)
        candidates = composite.reshape(rows, ng, 3, count).permute(
            0, 1, 3, 2
        )
        best_indices = torch.where(improve[:, :, :, None], candidates, best_indices)
    return best_error, best_state, best_indices, None


def _score_nvq1_banks(
    xvec: torch.Tensor,
    wvec: torch.Tensor,
    constant: torch.Tensor,
    anchor: torch.Tensor,
    rows: int,
    ng: int,
    pool: _Pool,
    start: int,
    stop: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    count = stop - start
    states = 1 << pool.spec.state_bits
    delta = 0.15625 if pool.spec is NEPQ1_S else 0.125
    best_error = torch.full(
        (rows, ng, count), torch.inf, device=xvec.device, dtype=torch.float32
    )
    best_state = torch.zeros((rows, ng, count), device=xvec.device, dtype=torch.uint8)
    best_aux = torch.zeros_like(best_state)
    best_indices = torch.zeros(
        (rows, ng, count, 3), device=xvec.device, dtype=torch.int64
    )
    weighted_x = wvec * xvec
    group_anchor = anchor.repeat_interleave(ng * 3)
    for aux in range(2):
        base = (
            pool.codes[start:stop, aux]
            if pool.spec is NEPQ1_S
            else pool.codes[start:stop]
        )
        shifted = base + (delta if aux == 0 else -delta)
        flat = shifted.reshape(-1, 8)
        cross = (weighted_x @ flat.T).reshape(-1, count, shifted.shape[1])
        quad = (wvec @ flat.square().T).reshape_as(cross)
        for state in range(states):
            scale = group_anchor[:, None] * float(state)
            variable = scale[:, :, None].square() * quad - 2.0 * scale[:, :, None] * cross
            minimum, indices = variable.min(2)
            error = (constant[:, None] + minimum).reshape(rows, ng, 3, count).sum(2)
            improve = error < best_error
            best_error = torch.where(improve, error, best_error)
            best_state = torch.where(
                improve, torch.full_like(best_state, state), best_state
            )
            best_aux = torch.where(
                improve, torch.full_like(best_aux, aux), best_aux
            )
            candidates = indices.reshape(rows, ng, 3, count).permute(0, 1, 3, 2)
            best_indices = torch.where(
                improve[:, :, :, None], candidates, best_indices
            )
    return best_error, best_state, best_indices, best_aux


def _select_assignment(
    value: torch.Tensor,
    objective: torch.Tensor,
    anchor: torch.Tensor,
    pool: _Pool,
    bank_chunk: int,
) -> _Assignment:
    if value.device.type == "mps" and pool.spec in {NEPQ0_S, NEPQ0_L}:
        from mfq.quantize.metal.nepq0_pool import assign_nepq0_pool

        assert pool.scale is not None
        assert pool.first_codes is not None
        assert pool.second_codes is not None
        bank_ids, state, indices, row_error = assign_nepq0_pool(
            value,
            objective,
            anchor,
            pool.scale,
            pool.first_codes,
            pool.second_codes,
        )
        return _Assignment(
            bank_ids=bank_ids,
            state=state,
            indices=indices,
            aux=None,
            anchor=anchor,
            row_error=row_error,
        )
    if value.device.type == "mps" and pool.spec in {NEPQ1_S, NEPQ1_L}:
        from mfq.quantize.metal.nepq1 import assign_nepq1

        assert pool.codes is not None
        bank_ids, state, indices, aux, row_error = assign_nepq1(
            value,
            objective,
            anchor,
            pool.codes,
            states=1 << pool.spec.state_bits,
            delta=0.15625 if pool.spec is NEPQ1_S else 0.125,
            banked_codebooks=pool.spec is NEPQ1_S,
        )
        return _Assignment(
            bank_ids=bank_ids,
            state=state,
            indices=indices,
            aux=aux,
            anchor=anchor,
            row_error=row_error,
        )
    rows, width = value.shape
    ng = math.ceil(width / 24)
    pad = ng * 24 - width
    if pad:
        value = torch.nn.functional.pad(value, (0, pad))
        objective = torch.nn.functional.pad(objective, (0, pad))
    xvec = value.reshape(rows * ng * 3, 8)
    wvec = objective.reshape_as(xvec)
    constant = (wvec * xvec.square()).sum(1)
    nsuper = math.ceil(ng / 4)
    selected_error = torch.full(
        (rows, nsuper), torch.inf, device=value.device, dtype=torch.float32
    )
    selected_bank = torch.zeros((rows, nsuper), device=value.device, dtype=torch.uint8)
    selected_state = torch.zeros((rows, ng), device=value.device, dtype=torch.uint8)
    selected_indices = torch.zeros(
        (rows, ng, 3), device=value.device, dtype=torch.int64
    )
    selected_aux = (
        torch.zeros((rows, ng), device=value.device, dtype=torch.uint8)
        if pool.spec.aux_bits
        else None
    )
    scorer = (
        _score_npq_banks
        if pool.spec is NEPQ0_S or pool.spec is NEPQ0_L
        else _score_nvq1_banks
    )
    for start in range(0, pool.banks, bank_chunk):
        stop = min(start + bank_chunk, pool.banks)
        error, state, indices, aux = scorer(
            xvec, wvec, constant, anchor, rows, ng, pool, start, stop
        )
        if ng % 4:
            error_for_sum = torch.nn.functional.pad(error, (0, 0, 0, 4 - ng % 4))
        else:
            error_for_sum = error
        super_error = error_for_sum.reshape(rows, nsuper, 4, stop - start).sum(2)
        candidate_error, local_bank = super_error.min(2)
        improve = candidate_error < selected_error
        selected_error = torch.where(improve, candidate_error, selected_error)
        selected_bank = torch.where(
            improve, (local_bank + start).to(torch.uint8), selected_bank
        )
        group_local = local_bank.repeat_interleave(4, dim=1)[:, :ng]
        chosen_state = torch.gather(state, 2, group_local[:, :, None]).squeeze(2)
        chosen_indices = torch.gather(
            indices, 2, group_local[:, :, None, None].expand(rows, ng, 1, 3)
        ).squeeze(2)
        group_improve = improve.repeat_interleave(4, dim=1)[:, :ng]
        selected_state = torch.where(group_improve, chosen_state, selected_state)
        selected_indices = torch.where(
            group_improve[:, :, None], chosen_indices, selected_indices
        )
        if selected_aux is not None and aux is not None:
            chosen_aux = torch.gather(aux, 2, group_local[:, :, None]).squeeze(2)
            selected_aux = torch.where(group_improve, chosen_aux, selected_aux)
    return _Assignment(
        bank_ids=selected_bank,
        state=selected_state,
        indices=selected_indices,
        aux=selected_aux,
        anchor=anchor,
        row_error=selected_error.sum(1),
    )


def _basis(assignment: _Assignment, pool: _Pool, width: int) -> torch.Tensor:
    rows, ng = assignment.state.shape
    group_bank = assignment.bank_ids.repeat_interleave(4, dim=1)[:, :ng].long()
    state = assignment.state.long()
    indices = assignment.indices.long()
    if pool.spec is NEPQ0_S or pool.spec is NEPQ0_L:
        assert pool.first_codes is not None and pool.second_codes is not None
        first_index = indices & 7
        second_index = indices >> 3
        first = pool.first_codes[
            group_bank[:, :, None], state[:, :, None], first_index
        ]
        second = pool.second_codes[
            group_bank[:, :, None], state[:, :, None], second_index
        ]
        code = torch.cat((first, second), dim=3)
        relative = pool.scale[group_bank, state]
        result = code * relative[:, :, None, None]
    else:
        assert pool.codes is not None
        aux = assignment.aux.long()
        if pool.spec is NEPQ1_S:
            code = pool.codes[group_bank[:, :, None], aux[:, :, None], indices]
            delta = torch.where(aux == 0, 0.15625, -0.15625)
        else:
            code = pool.codes[group_bank[:, :, None], indices]
            delta = torch.where(aux == 0, 0.125, -0.125)
        result = state[:, :, None, None] * (
            code + delta[:, :, None, None]
        )
    return result.reshape(rows, ng * 24)[:, :width]


def _project(
    value: torch.Tensor,
    objective: torch.Tensor,
    base_anchor: torch.Tensor,
    pool: _Pool,
    config: NepqQuantConfig,
) -> _Assignment:
    best: _Assignment | None = None
    for multiplier in config.anchor_multipliers:
        anchor = (base_anchor * float(multiplier)).to(torch.float16).to(torch.float32)
        assignment: _Assignment | None = None
        for refine in range(config.refine_steps + 1):
            assignment = _select_assignment(
                value, objective, anchor, pool, config.bank_chunk
            )
            basis = _basis(assignment, pool, value.shape[1])
            if refine < config.refine_steps:
                numerator = (objective * value * basis).sum(1)
                denominator = (objective * basis.square()).sum(1)
                anchor = torch.where(
                    denominator > 0, numerator / denominator, anchor
                ).clamp_min(0).to(torch.float16).to(torch.float32)
        assert assignment is not None
        assignment.anchor = anchor
        basis = _basis(assignment, pool, value.shape[1])
        assignment.row_error = (
            objective * (value - anchor[:, None] * basis).square()
        ).sum(1)
        if best is None:
            best = assignment
            continue
        improve = assignment.row_error < best.row_error
        best.bank_ids = torch.where(improve[:, None], assignment.bank_ids, best.bank_ids)
        best.state = torch.where(improve[:, None], assignment.state, best.state)
        best.indices = torch.where(improve[:, None, None], assignment.indices, best.indices)
        if best.aux is not None:
            best.aux = torch.where(improve[:, None], assignment.aux, best.aux)
        best.anchor = torch.where(improve, assignment.anchor, best.anchor)
        best.row_error = torch.where(improve, assignment.row_error, best.row_error)
    assert best is not None
    return best


def _can_use_native_nepq0_s(
    value: torch.Tensor,
    importance,
    pool: _Pool,
    config: NepqQuantConfig,
) -> bool:
    return (
        pool.spec is NEPQ0_S
        and (value.is_cuda or value.device.type == "mps")
        and importance is None
        and pool.banks == 256
        and config.anchor_multipliers == (1.0,)
        and config.refine_steps == 1
        and pool.native_scale is not None
        and pool.native_first_codes is not None
        and pool.native_second_codes is not None
    )


def _project_native_nepq0_s(
    value: torch.Tensor,
    base_anchor: torch.Tensor,
    pool: _Pool,
) -> _Assignment:
    assert pool.native_scale is not None
    assert pool.native_first_codes is not None
    assert pool.native_second_codes is not None
    initial_anchor = base_anchor.to(torch.float16).to(torch.float32).contiguous()
    if value.is_cuda:
        from mfq.quantize.cuda._ext import ext

        assign = ext().nepq0_s_assign
    elif value.device.type == "mps":
        from mfq.quantize.metal.nepq import nepq0_s_assign

        assign = nepq0_s_assign
    else:
        raise RuntimeError("native NEPQ0-S assignment requires CUDA or MPS")
    anchor, bank_ids, state, indices = assign(
        value,
        initial_anchor,
        pool.native_scale,
        pool.native_first_codes,
        pool.native_second_codes,
    )
    return _Assignment(
        bank_ids=bank_ids,
        state=state,
        indices=indices,
        aux=None,
        anchor=anchor,
        row_error=torch.zeros_like(anchor),
    )


def _clone_assignment(value: _Assignment) -> _Assignment:
    return _Assignment(
        bank_ids=value.bank_ids.clone(),
        state=value.state.clone(),
        indices=value.indices.clone(),
        aux=None if value.aux is None else value.aux.clone(),
        anchor=value.anchor.clone(),
        row_error=value.row_error.clone(),
    )


def _replace_better(
    best: _Assignment,
    candidate: _Assignment,
    better: torch.Tensor,
) -> None:
    best.bank_ids = torch.where(
        better[:, None], candidate.bank_ids, best.bank_ids
    )
    best.state = torch.where(better[:, None], candidate.state, best.state)
    best.indices = torch.where(
        better[:, None, None], candidate.indices, best.indices
    )
    if best.aux is not None and candidate.aux is not None:
        best.aux = torch.where(better[:, None], candidate.aux, best.aux)
    best.anchor = torch.where(better, candidate.anchor, best.anchor)
    best.row_error = torch.where(
        better, candidate.row_error, best.row_error
    )


def _reconstruct(
    assignment: _Assignment,
    pool: _Pool,
    width: int,
) -> torch.Tensor:
    return assignment.anchor[:, None] * _basis(assignment, pool, width)


def _rotated_imatrix_error(
    value: torch.Tensor,
    reconstruction: torch.Tensor,
    objective: torch.Tensor,
    block: int,
) -> torch.Tensor:
    original_residual = _fwht_blocks(value - reconstruction, block)
    return (objective * original_residual.square()).sum(1)


def _native_admm_projector_available(
    value: torch.Tensor,
    pool: _Pool,
) -> bool:
    return (
        pool.spec is NEPQ0_S
        and (value.is_cuda or value.device.type == "mps")
        and pool.banks == 256
        and pool.native_scale is not None
        and pool.native_first_codes is not None
        and pool.native_second_codes is not None
    )


def _project_hadamard_imatrix_admm(
    value: torch.Tensor,
    objective: torch.Tensor,
    base_anchor: torch.Tensor,
    pool: _Pool,
    config: NepqQuantConfig,
    block: int,
) -> _Assignment:
    if config.admm_iterations <= 0:
        raise ValueError("H2048 imatrix quantization requires ADMM iterations")
    row_mean = objective.mean(1, keepdim=True)
    if torch.any(row_mean <= 0):
        raise ValueError("every NEPQ ADMM row needs positive imatrix mass")
    solve_h = objective / row_mean
    native = _native_admm_projector_available(value, pool)
    q_config = NepqQuantConfig(
        anchor_multipliers=(1.0,),
        refine_steps=1,
        row_chunk=config.row_chunk,
        bank_chunk=config.bank_chunk,
    )
    if native:
        assignment = _project_native_nepq0_s(value, base_anchor, pool)
    else:
        assignment = _project(
            value, torch.ones_like(value), base_anchor, pool, q_config
        )
    q = _reconstruct(assignment, pool, value.shape[1])
    assignment.row_error = _rotated_imatrix_error(
        value, q, solve_h, block
    )
    best = _clone_assignment(assignment)
    x_eigenbasis = _fwht_blocks(value, block)
    u = torch.zeros_like(value)

    for _ in range(config.admm_iterations):
        v_eigenbasis = _fwht_blocks(q - u, block)
        z_eigenbasis = (
            solve_h * x_eigenbasis
            + config.admm_rho * v_eigenbasis
        ) / (solve_h + config.admm_rho)
        z = _fwht_blocks(z_eigenbasis, block)
        target = z + u
        if native:
            assignment = _project_native_nepq0_s(
                target, assignment.anchor, pool
            )
        else:
            assignment = _project(
                target,
                torch.ones_like(target),
                assignment.anchor,
                pool,
                q_config,
            )
        q = _reconstruct(assignment, pool, value.shape[1])
        u.add_(z - q)
        assignment.row_error = _rotated_imatrix_error(
            value, q, solve_h, block
        )
        _replace_better(
            best, assignment, assignment.row_error < best.row_error
        )
    return best


@torch.inference_mode()
def quantize_nepq_fixed(
    weight,
    spec: str | int | NepqSpec,
    table_payloads,
    *,
    importance=None,
    initial_anchor=None,
    rotation_block: int = 0,
    rotation_seed: int = 0,
    config: NepqQuantConfig | None = None,
    device: str | torch.device = "cuda",
    progress: Callable[[int, int], None] | None = None,
) -> NepqTensor:
    """Quantize ``[experts,out,K]`` against a frozen cross-expert table pool."""

    spec = nepq_spec(spec)
    config = NepqQuantConfig() if config is None else config
    shape = _source_shape(weight)
    experts, out_per_expert, width = shape
    if rotation_block:
        if rotation_block & (rotation_block - 1) or width % rotation_block:
            raise ValueError("NEPQ rotation block must be a power of two dividing K")
        if not 0 <= rotation_seed < 1 << 64:
            raise ValueError("NEPQ rotation seed must fit uint64")
        if importance is not None and config.admm_iterations <= 0:
            raise ValueError(
                "rotated NEPQ with diagonal importance requires the ADMM calibration path"
            )
    elif rotation_seed:
        raise ValueError("NEPQ rotation seed requires a nonzero block")
    elif config.admm_iterations:
        raise ValueError("NEPQ ADMM iterations require a Hadamard rotation")
    tables = _table_array(spec, table_payloads)
    pool = _decode_pool(spec, tables, device)
    rows = experts * out_per_expert
    ng = math.ceil(width / 24)
    nvec = width // 8
    nsuper = math.ceil(ng / 4)
    anchors = np.empty(rows, dtype=np.float32)
    states = np.empty((rows, ng), dtype=np.uint8)
    index_dtype = np.uint8 if spec.index_bits <= 8 else np.uint16
    indices = np.empty((rows, nvec), dtype=index_dtype)
    aux = np.empty((rows, ng), dtype=np.uint8) if spec.aux_bits else None
    bank_ids = np.empty((rows, nsuper), dtype=np.uint8)
    initial = None
    if initial_anchor is not None:
        initial = np.asarray(initial_anchor, dtype=np.float32)
        if initial.shape != shape[:2]:
            raise ValueError(f"NEPQ initial_anchor must have shape {shape[:2]}")
        if not np.isfinite(initial).all() or np.any(initial < 0):
            raise ValueError("NEPQ initial anchors must be finite and non-negative")
        initial = initial.reshape(-1)
    signs = (
        torch.as_tensor(
            rotation_signs(width, rotation_block, rotation_seed),
            device=device,
            dtype=torch.float32,
        )
        if rotation_block
        else None
    )

    for start in range(0, rows, config.row_chunk):
        stop = min(start + config.row_chunk, rows)
        value = _slice_rows(weight, start, stop, width, device)
        if signs is not None:
            value = _fwht_blocks(value * signs, rotation_block)
        if initial is None:
            base_anchor = value.abs().amax(1) / pool.maximum_basis
        else:
            base_anchor = torch.as_tensor(
                initial[start:stop], device=device, dtype=torch.float32
            )
        if signs is not None and importance is not None:
            objective = _importance_slice(
                importance, start, stop, shape, device
            )
            assignment = _project_hadamard_imatrix_admm(
                value,
                objective,
                base_anchor,
                pool,
                config,
                rotation_block,
            )
        elif _can_use_native_nepq0_s(value, importance, pool, config):
            assignment = _project_native_nepq0_s(value, base_anchor, pool)
        else:
            objective = _importance_slice(
                importance, start, stop, shape, device
            )
            assignment = _project(value, objective, base_anchor, pool, config)
        count = stop - start
        anchors[start:stop] = assignment.anchor.cpu().numpy()
        states[start:stop] = assignment.state.cpu().numpy()
        indices[start:stop] = (
            assignment.indices.reshape(count, ng * 3)[:, :nvec].cpu().numpy()
        )
        bank_ids[start:stop] = assignment.bank_ids.cpu().numpy()
        if aux is not None:
            aux[start:stop] = assignment.aux.cpu().numpy()
        if progress is not None:
            progress(stop, rows)

    result = NepqTensor(
        spec=spec,
        shape=shape,
        neuron_scale=anchors.reshape(experts, out_per_expert),
        state=states.reshape(experts, out_per_expert, ng),
        indices=indices.reshape(experts, out_per_expert, nvec),
        aux=(aux.reshape(experts, out_per_expert, ng) if aux is not None else None),
        bank_ids=bank_ids.reshape(experts, out_per_expert, nsuper),
        table_payloads=tables,
        rotation_block=int(rotation_block),
        rotation_seed=int(rotation_seed),
    )
    validate_nepq(result)
    return result


__all__ = ["NepqQuantConfig", "quantize_nepq_fixed"]
