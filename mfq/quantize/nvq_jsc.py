"""NVQ joint scale/codebook-state quantizer for E8 and D4 base layouts.

The deployment profile keeps one 4-bit state per gs24 group and one 7-bit
even-parity sign mask per 8 weights.  NVQ2J stores one E8 index per 8 weights;
NVQ3J stores one D4 index per 4 weights.  E8 codebooks support 256, 1024, or
4096 entries; D4 codebooks support 256, 512, or 1024 entries.  FP32 codebooks
remain available only as an offline oracle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from mfq.formats.nvq import NVQ2_E8, NvqJscTensor, NvqSpec, codebook_for
from mfq.quantize.nvq_quant_torch import (
    _encode_even_parity_signs,
    _fp16_round,
    _pad_weight,
    _prepare_weight,
    _reassign_nvq,
    _search_nvq_groups,
)


_GROUP_SIZE = 24
_STATE_COUNT = 16
_METAL_E8_ANCHOR_GROUPS = 8
_METAL_E8_EXTRA_ITERATIONS = 2


@dataclass(frozen=True)
class NvqJscConfig:
    banks: int = 1
    iterations: int = 4
    assignment_refine_steps: int = 2
    search_steps: int = 19
    raw_multiplier: int = 8
    learned_scale_lut: bool = True
    codebook_storage: str = "int8"
    group_chunk: int = 1024
    seed: int = 20260717
    spec: NvqSpec = NVQ2_E8

    def __post_init__(self) -> None:
        if self.banks not in {1, 2, 4}:
            raise ValueError("NVQ-JSC banks must be 1, 2, or 4")
        if self.iterations < 0 or self.assignment_refine_steps < 0:
            raise ValueError("NVQ-JSC iteration counts must be non-negative")
        if self.search_steps <= 0:
            raise ValueError("NVQ-JSC search_steps must be positive")
        if not 1 <= self.raw_multiplier <= 16:
            raise ValueError("NVQ-JSC raw_multiplier must be in [1, 16]")
        if (
            self.spec.groupsize != _GROUP_SIZE
            or self.spec.sub_bits != 4
            or self.spec.sign_mode != "even"
        ):
            raise ValueError("NVQ-JSC requires gs24, a 4-bit state, and parity signs")
        if int(codebook_for(self.spec).max()) * self.raw_multiplier > 127:
            raise ValueError("NVQ-JSC initial codebook exceeds int8 magnitude range")
        if self.codebook_storage not in {"int8", "float32"}:
            raise ValueError("NVQ-JSC codebook_storage must be int8 or float32")
        if self.group_chunk <= 0:
            raise ValueError("NVQ-JSC group_chunk must be positive")


@dataclass(frozen=True)
class NvqJscIteration:
    iteration: int
    weighted_sse: float
    weighted_nmse_percent: float
    used_states: int
    used_banks: int
    used_codes: tuple[int, ...]


@dataclass(frozen=True)
class NvqJscTables:
    scale_lut: np.ndarray
    bank_for_state: np.ndarray
    codebooks: np.ndarray
    spec: NvqSpec = NVQ2_E8


def _validate_codebooks(
    codebooks: np.ndarray,
    banks: int,
    storage: str,
    spec: NvqSpec,
) -> np.ndarray:
    value = np.asarray(codebooks)
    expected = (banks, spec.codebook_entries, spec.vector_size)
    if value.shape != expected:
        raise ValueError(f"codebooks have shape {value.shape}, expected {expected}")
    if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 127):
        raise ValueError("codebook coordinates must be finite and in [0, 127]")
    if storage == "int8":
        rounded = np.rint(value)
        if not np.array_equal(value, rounded):
            raise ValueError("int8 codebook coordinates must be integers")
        result = np.ascontiguousarray(rounded, dtype=np.int8)
    elif storage == "float32":
        result = np.ascontiguousarray(value, dtype=np.float32)
    else:
        raise ValueError(f"unsupported codebook storage: {storage}")
    if np.any(np.all(result == 0, axis=-1)):
        raise ValueError("codewords must not be all zero")
    return result


def initial_raw_codebooks(config: NvqJscConfig) -> np.ndarray:
    """Build deterministic, slightly diversified raw-int8 initial banks."""

    base = codebook_for(config.spec).astype(np.int16) * config.raw_multiplier
    result = np.repeat(base[None, :, :], config.banks, axis=0)
    if config.banks > 1:
        rng = np.random.default_rng(config.seed)
        for bank in range(1, config.banks):
            jitter = rng.integers(-bank, bank + 1, size=base.shape, dtype=np.int16)
            candidate = np.clip(base + jitter, 0, 127)
            zero = np.all(candidate == 0, axis=1)
            candidate[zero] = base[zero]
            result[bank] = candidate
    return _validate_codebooks(result, config.banks, "int8", config.spec)


def _initial_state_tables(config: NvqJscConfig) -> tuple[torch.Tensor, torch.Tensor]:
    bank = torch.arange(_STATE_COUNT, dtype=torch.int64) % config.banks
    if config.banks == 1:
        alpha = torch.arange(_STATE_COUNT, dtype=torch.float32)
    else:
        rank = torch.div(torch.arange(_STATE_COUNT), config.banks, rounding_mode="floor")
        levels = _STATE_COUNT // config.banks
        if config.spec.vector_size == 4:
            alpha = rank.to(torch.float32) + 1.0
        else:
            alpha = 15.0 * (rank.to(torch.float32) + 1.0) / float(levels)
    return alpha, bank


def initial_jsc_tables(config: NvqJscConfig = NvqJscConfig()) -> NvqJscTables:
    """Return deterministic weight-only tables before tensor-wise fitting."""

    alpha, bank = _initial_state_tables(config)
    return NvqJscTables(
        scale_lut=alpha.numpy(),
        bank_for_state=bank.numpy().astype(np.uint8, copy=False),
        codebooks=initial_raw_codebooks(config),
        spec=config.spec,
    )


def _native_search(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    codebook: torch.Tensor,
    ng: int,
    valid_last: int,
    search_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from mfq.quantize.cuda._ext import ext

    return tuple(
        ext().nvq_search(
            xgroup,
            wgroup,
            codebook,
            ng,
            valid_last,
            int(codebook.shape[1]),
            search_steps,
            float(codebook.max().item()),
        )
    )


def _metal_search(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    codebook: torch.Tensor,
    ng: int,
    valid_last: int,
    search_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from mfq.quantize.metal.nvq import nvq_search

    return nvq_search(
        xgroup,
        wgroup,
        codebook,
        ng,
        valid_last,
        search_steps,
        float(codebook.max().item()),
    )


def _native_reassign(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    scale: torch.Tensor,
    codebook: torch.Tensor,
    ng: int,
    valid_last: int,
) -> torch.Tensor:
    from mfq.quantize.cuda._ext import ext

    return ext().nvq_reassign(
        xgroup,
        wgroup,
        scale.contiguous(),
        codebook,
        ng,
        valid_last,
        int(codebook.shape[1]),
    )


def _metal_reassign(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    scale: torch.Tensor,
    codebook: torch.Tensor,
    ng: int,
    valid_last: int,
) -> torch.Tensor:
    from mfq.quantize.metal.nvq import nvq_reassign

    return nvq_reassign(
        xgroup,
        wgroup,
        scale,
        codebook,
        ng,
        valid_last,
    )


def _search(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    codebook: torch.Tensor,
    ng: int,
    valid_last: int,
    config: NvqJscConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if codebook.dtype == torch.int8 and codebook.is_cuda:
        return _native_search(
            xgroup,
            wgroup,
            codebook,
            ng,
            valid_last,
            config.search_steps,
        )
    if codebook.dtype == torch.int8 and codebook.device.type == "mps":
        return _metal_search(
            xgroup,
            wgroup,
            codebook,
            ng,
            valid_last,
            config.search_steps,
        )
    return _search_nvq_groups(
        xgroup,
        wgroup,
        config.spec,
        codebook.to(torch.float32),
        search_steps=config.search_steps,
        group_chunk=config.group_chunk,
    )


def _reassign(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    scale: torch.Tensor,
    codebook: torch.Tensor,
    ng: int,
    valid_last: int,
    config: NvqJscConfig,
) -> torch.Tensor:
    if codebook.dtype == torch.int8 and codebook.is_cuda:
        return _native_reassign(
            xgroup,
            wgroup,
            scale,
            codebook,
            ng,
            valid_last,
        )
    if codebook.dtype == torch.int8 and codebook.device.type == "mps":
        return _metal_reassign(
            xgroup,
            wgroup,
            scale,
            codebook,
            ng,
            valid_last,
        )
    return _reassign_nvq(
        xgroup,
        wgroup,
        scale,
        config.spec,
        codebook.to(torch.float32),
        group_chunk=config.group_chunk,
    )


def _nearest_state(
    ratio: torch.Tensor,
    alpha: torch.Tensor,
    bank_for_state: torch.Tensor,
    bank: int,
) -> torch.Tensor:
    distance = (ratio.unsqueeze(1) - alpha.unsqueeze(0)).abs()
    distance = torch.where(
        bank_for_state.unsqueeze(0) == bank,
        distance,
        torch.full_like(distance, torch.inf),
    )
    return distance.argmin(dim=1)


def _codes_for_assignment(
    codebooks: torch.Tensor,
    bank: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    vector_size = int(codebooks.shape[2])
    entries = int(codebooks.shape[1])
    flat_codebooks = codebooks.reshape(-1, vector_size)
    keys = bank.unsqueeze(1) * entries + indices
    return flat_codebooks[keys].reshape(-1, _GROUP_SIZE).to(torch.float32)


def _native_e8_jsc_assignment_supported(
    codebooks: torch.Tensor,
    config: NvqJscConfig,
) -> bool:
    return (
        codebooks.is_cuda
        and codebooks.dtype == torch.int8
        and config.banks == 4
        and config.spec.vector_size == 8
        and config.spec.codebook_entries in {1024, 4096}
        and tuple(codebooks.shape)
        == (
            4,
            config.spec.codebook_entries,
            config.spec.vector_size,
        )
    )


def _metal_e8_jsc_assignment_supported(
    codebooks: torch.Tensor,
    config: NvqJscConfig,
) -> bool:
    return (
        codebooks.device.type == "mps"
        and codebooks.dtype == torch.int8
        and config.banks == 4
        and config.spec.vector_size == 8
        and config.spec.codebook_entries in {256, 1024, 4096}
        and tuple(codebooks.shape)
        == (
            4,
            config.spec.codebook_entries,
            config.spec.vector_size,
        )
    )


def _native_e8_search_banks(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    ng: int,
    valid_last: int,
    search_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from mfq.quantize.cuda._ext import ext

    bank_qmax = codebooks.amax((1, 2)).to(torch.float32).contiguous()
    scales, indices = ext().nvq2j_search_banks(
        xgroup,
        wgroup,
        codebooks,
        bank_qmax,
        ng,
        valid_last,
        search_steps,
    )
    return scales, indices.to(torch.int64)


def _assign_groups(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    codebooks: torch.Tensor,
    neuron_scale: torch.Tensor,
    alpha: torch.Tensor,
    bank_for_state: torch.Tensor,
    *,
    out: int,
    ng: int,
    valid_last: int,
    config: NvqJscConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if _metal_e8_jsc_assignment_supported(codebooks, config):
        from mfq.quantize.metal.nvq_jsc import nvq2j_assign

        padded_width = ng * _GROUP_SIZE
        valid_width = (ng - 1) * _GROUP_SIZE + valid_last
        state_2d, indices_3d = nvq2j_assign(
            xgroup.reshape(out, padded_width),
            wgroup.reshape(out, padded_width),
            neuron_scale,
            alpha,
            bank_for_state,
            codebooks,
            valid_width,
        )
        state = state_2d.reshape(-1).to(torch.int64)
        indices = indices_3d.reshape(
            -1, _GROUP_SIZE // config.spec.vector_size
        ).to(torch.int64)
        bank = bank_for_state[state]
        code = _codes_for_assignment(codebooks, bank, indices)
        scale = neuron_scale.repeat_interleave(ng) * alpha[state]
        error = (
            wgroup * (scale.unsqueeze(1) * code - xgroup).square()
        ).sum(1)
        return state, bank, indices, error

    if _native_e8_jsc_assignment_supported(codebooks, config):
        raw_scales, _ = _native_e8_search_banks(
            xgroup,
            wgroup,
            codebooks,
            ng=ng,
            valid_last=valid_last,
            search_steps=config.search_steps,
        )
        group_anchor = neuron_scale.repeat_interleave(ng)
        safe_anchor = torch.where(
            group_anchor > 0,
            group_anchor,
            torch.ones_like(group_anchor),
        )
        candidate_state = torch.empty(
            (group_anchor.numel(), config.banks),
            device=xgroup.device,
            dtype=torch.int64,
        )
        candidate_scale = torch.empty_like(raw_scales)
        for bank_id in range(config.banks):
            state = _nearest_state(
                raw_scales[:, bank_id] / safe_anchor,
                alpha,
                bank_for_state,
                bank_id,
            )
            candidate_state[:, bank_id] = state
            candidate_scale[:, bank_id] = group_anchor * alpha[state]
        candidate_indices = torch.stack(
            [
                _native_reassign(
                    xgroup,
                    wgroup,
                    candidate_scale[:, bank_id].contiguous(),
                    codebooks[bank_id],
                    ng,
                    valid_last,
                )
                for bank_id in range(config.banks)
            ],
            dim=1,
        )
        best_error = torch.full_like(group_anchor, torch.inf)
        best_state = torch.zeros_like(group_anchor, dtype=torch.int64)
        best_bank = torch.zeros_like(group_anchor, dtype=torch.int64)
        best_indices = torch.zeros(
            (group_anchor.numel(), _GROUP_SIZE // config.spec.vector_size),
            device=xgroup.device,
            dtype=torch.int64,
        )
        for bank_id in range(config.banks):
            indices = candidate_indices[:, bank_id]
            code = (
                codebooks[bank_id][indices]
                .reshape_as(xgroup)
                .to(torch.float32)
            )
            error = (
                wgroup
                * (
                    candidate_scale[:, bank_id].unsqueeze(1) * code
                    - xgroup
                ).square()
            ).sum(1)
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best_state = torch.where(
                better,
                candidate_state[:, bank_id],
                best_state,
            )
            best_bank = torch.where(
                better,
                torch.full_like(best_bank, bank_id),
                best_bank,
            )
            best_indices = torch.where(
                better.unsqueeze(-1), indices, best_indices
            )
        return best_state, best_bank, best_indices, best_error

    group_anchor = neuron_scale.repeat_interleave(ng)
    vectors_per_group = _GROUP_SIZE // int(codebooks.shape[2])
    safe_anchor = torch.where(group_anchor > 0, group_anchor, torch.ones_like(group_anchor))
    best_error = torch.full_like(group_anchor, torch.inf)
    best_state = torch.zeros_like(group_anchor, dtype=torch.int64)
    best_bank = torch.zeros_like(group_anchor, dtype=torch.int64)
    best_indices = torch.zeros(
        (group_anchor.numel(), vectors_per_group),
        device=xgroup.device,
        dtype=torch.int64,
    )

    for bank in range(codebooks.shape[0]):
        raw_scale, _ = _search(
            xgroup,
            wgroup,
            codebooks[bank],
            ng,
            valid_last,
            config,
        )
        state = _nearest_state(raw_scale / safe_anchor, alpha, bank_for_state, bank)
        effective_scale = group_anchor * alpha[state]
        indices = _reassign(
            xgroup,
            wgroup,
            effective_scale,
            codebooks[bank],
            ng,
            valid_last,
            config,
        )
        code = codebooks[bank][indices].reshape_as(xgroup).to(torch.float32)
        error = (wgroup * (effective_scale.unsqueeze(1) * code - xgroup).square()).sum(1)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_state = torch.where(better, state, best_state)
        best_bank = torch.where(
            better,
            torch.full_like(best_bank, bank),
            best_bank,
        )
        best_indices = torch.where(
            better.unsqueeze(-1), indices, best_indices
        )

    return best_state, best_bank, best_indices, best_error


def _refit_scale_tables(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    codebooks: torch.Tensor,
    state: torch.Tensor,
    bank: torch.Tensor,
    indices: torch.Tensor,
    neuron_scale: torch.Tensor,
    alpha: torch.Tensor,
    *,
    out: int,
    ng: int,
    learned_scale_lut: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    code = _codes_for_assignment(codebooks, bank, indices)
    numerator = (wgroup * xgroup * code).sum(1)
    curvature = (wgroup * code * code).sum(1)
    raw_scale = torch.where(
        curvature > 0,
        numerator / curvature,
        torch.zeros_like(numerator),
    ).clamp_min(0)

    alpha_group = alpha[state]
    num_anchor = (curvature * alpha_group * raw_scale).reshape(out, ng).sum(1)
    den_anchor = (curvature * alpha_group.square()).reshape(out, ng).sum(1)
    fitted_anchor = torch.where(
        den_anchor > 0,
        num_anchor / den_anchor,
        torch.zeros_like(num_anchor),
    ).clamp_min(0)
    fitted_anchor = _fp16_round(fitted_anchor)

    if not learned_scale_lut:
        return fitted_anchor, alpha

    group_anchor = fitted_anchor.repeat_interleave(ng)
    lut_num = torch.zeros_like(alpha)
    lut_den = torch.zeros_like(alpha)
    lut_num.scatter_add_(0, state, curvature * group_anchor * raw_scale)
    lut_den.scatter_add_(0, state, curvature * group_anchor.square())
    fitted_alpha = torch.where(lut_den > 0, lut_num / lut_den, alpha).clamp_min(0)
    maximum = fitted_alpha.max()
    if float(maximum.item()) > 0:
        factor = maximum / 15.0
        fitted_alpha = fitted_alpha / factor
        fitted_anchor = _fp16_round(fitted_anchor * factor)
    return fitted_anchor, _fp16_round(fitted_alpha)


def _update_codebooks(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    codebooks: torch.Tensor,
    state: torch.Tensor,
    bank: torch.Tensor,
    indices: torch.Tensor,
    neuron_scale: torch.Tensor,
    alpha: torch.Tensor,
    *,
    ng: int,
    storage: str,
) -> torch.Tensor:
    vector_size = int(codebooks.shape[2])
    vectors_per_group = _GROUP_SIZE // vector_size
    vectors = xgroup.reshape(-1, vector_size)
    vector_weight = wgroup.reshape_as(vectors)
    vector_index = indices.reshape(-1)
    vector_bank = bank.repeat_interleave(vectors_per_group)
    vector_scale = (
        neuron_scale.repeat_interleave(ng) * alpha[state]
    ).repeat_interleave(vectors_per_group)
    codebook_entries = int(codebooks.shape[1])
    key = vector_bank * codebook_entries + vector_index
    entries = codebooks.shape[0] * codebook_entries
    numerator = torch.zeros(
        (entries, vector_size), device=xgroup.device, dtype=torch.float32
    )
    denominator = torch.zeros_like(numerator)
    expanded_key = key.unsqueeze(1).expand(-1, vector_size)
    numerator.scatter_add_(
        0,
        expanded_key,
        vector_weight * vector_scale.unsqueeze(1) * vectors,
    )
    denominator.scatter_add_(
        0,
        expanded_key,
        vector_weight * vector_scale.square().unsqueeze(1),
    )
    previous = codebooks.reshape(entries, vector_size).to(torch.float32)
    centroid = torch.where(denominator > 0, numerator / denominator, previous)
    if storage == "int8":
        updated = centroid.round().clamp(0, 127).to(torch.int8)
    elif storage == "float32":
        updated = centroid.clamp(0, 127)
    else:
        raise ValueError(f"unsupported codebook storage: {storage}")
    all_zero = torch.all(updated == 0, dim=1)
    updated[all_zero] = codebooks.reshape(entries, vector_size)[all_zero]
    return updated.reshape_as(codebooks).contiguous()


def _assignment_error(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    codebooks: torch.Tensor,
    state: torch.Tensor,
    bank: torch.Tensor,
    indices: torch.Tensor,
    neuron_scale: torch.Tensor,
    alpha: torch.Tensor,
    *,
    ng: int,
) -> torch.Tensor:
    code = _codes_for_assignment(codebooks, bank, indices)
    scale = neuron_scale.repeat_interleave(ng) * alpha[state]
    return (wgroup * (scale.unsqueeze(1) * code - xgroup).square()).sum()


def _validate_tables(
    tables: NvqJscTables,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, NvqSpec]:
    raw_codebooks = np.asarray(tables.codebooks)
    if raw_codebooks.ndim != 3:
        raise ValueError("NVQ-JSC codebooks must be a rank-3 array")
    codebooks = _validate_codebooks(
        raw_codebooks,
        int(raw_codebooks.shape[0]),
        "int8",
        tables.spec,
    )
    alpha = np.asarray(tables.scale_lut, dtype=np.float32).reshape(-1)
    bank = np.asarray(tables.bank_for_state).reshape(-1)
    if alpha.shape != (_STATE_COUNT,) or not np.isfinite(alpha).all() or np.any(alpha < 0):
        raise ValueError("NVQ-JSC scale_lut must contain 16 finite non-negative values")
    if float(alpha.max()) <= 0:
        raise ValueError("NVQ-JSC scale_lut must contain a positive value")
    if bank.shape != (_STATE_COUNT,) or not np.issubdtype(bank.dtype, np.integer):
        raise ValueError("NVQ-JSC bank_for_state must contain 16 integers")
    bank = np.ascontiguousarray(bank, dtype=np.uint8)
    if np.any(bank >= codebooks.shape[0]):
        raise ValueError("NVQ-JSC bank_for_state references a missing bank")
    return np.ascontiguousarray(alpha), bank, codebooks, tables.spec


@torch.inference_mode()
def quantize_nvq_jsc_fixed(
    weight: torch.Tensor,
    tables: NvqJscTables,
    *,
    importance: np.ndarray | torch.Tensor | None = None,
    assignment_refine_steps: int = 2,
    search_steps: int = 19,
    group_chunk: int = 1024,
    device: str | torch.device = "cuda",
) -> NvqJscTensor:
    """Quantize rows with one fixed tensor-wise JSC table set."""

    if assignment_refine_steps < 0 or search_steps <= 0 or group_chunk <= 0:
        raise ValueError("invalid NVQ-JSC fixed-assignment configuration")
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("NVQ-JSC CUDA quantization requested without a CUDA device")
    alpha_np, bank_np, codebooks_np, spec = _validate_tables(tables)
    value, out, neuron_len = _prepare_weight(weight, device)
    value, objective_weight, ng = _pad_weight(value, _GROUP_SIZE, importance)
    target, signs = _encode_even_parity_signs(value, objective_weight)
    xgroup = target.reshape(out * ng, _GROUP_SIZE).contiguous()
    wgroup = objective_weight.reshape_as(xgroup).contiguous()
    valid_last = neuron_len - (ng - 1) * _GROUP_SIZE
    codebook_dtype = (
        torch.int8
        if value.is_cuda or value.device.type == "mps"
        else torch.float32
    )
    codebooks = torch.as_tensor(
        codebooks_np, device=value.device, dtype=codebook_dtype
    ).contiguous()
    alpha = torch.as_tensor(alpha_np, device=value.device, dtype=torch.float32)
    bank_for_state = torch.as_tensor(bank_np, device=value.device, dtype=torch.int64)
    native_assign = None
    if value.is_cuda and codebooks.dtype == torch.int8:
        from mfq.quantize.cuda._ext import ext

        bank_count = int(codebooks_np.shape[0])
        balanced_banks = np.array_equal(
            np.bincount(bank_np, minlength=bank_count),
            np.full(bank_count, _STATE_COUNT // bank_count),
        )
        if (
            spec.vector_size == 8
            and spec.codebook_entries in {256, 1024, 4096}
            and bank_count == 4
            and balanced_banks
        ):
            native_assign = ext().nvq2j_assign
        elif (
            spec.vector_size == 4
            and spec.codebook_entries == 256
            and bank_count == 2
            and balanced_banks
        ):
            native_assign = ext().nvq3j_assign

    metal_assign = None
    if value.device.type == "mps":
        bank_count = int(codebooks_np.shape[0])
        balanced_banks = np.array_equal(
            np.bincount(bank_np, minlength=bank_count),
            np.full(bank_count, _STATE_COUNT // bank_count),
        )
        if (
            spec.vector_size == 8
            and spec.codebook_entries in {256, 1024, 4096}
            and bank_count == 4
            and balanced_banks
        ):
            from mfq.quantize.metal.nvq_jsc import nvq2j_assign

            metal_assign = nvq2j_assign
        elif (
            spec.vector_size == 4
            and spec.codebook_entries in {256, 512, 1024}
            and bank_count == 2
            and balanced_banks
        ):
            from mfq.quantize.metal.nvq_jsc import nvq3j_assign

            metal_assign = nvq3j_assign

    if native_assign is not None:
        bank_u8 = bank_for_state.to(torch.uint8).contiguous()
        native_codebooks = codebooks.permute(0, 2, 1).contiguous()
        bank_peak = codebooks.to(torch.float32).amax((1, 2))
        maximum_basis = (
            alpha * bank_peak[bank_for_state]
        ).amax().clamp_min(1e-20)
        weighted_target = torch.where(
            objective_weight > 0,
            target,
            torch.zeros((), device=target.device, dtype=target.dtype),
        )
        initial_anchor = _fp16_round(
            weighted_target[:, :neuron_len].amax(1) / maximum_basis
        ).contiguous()
        neuron_scale, state, indices = native_assign(
            target.contiguous(),
            objective_weight.contiguous(),
            initial_anchor,
            alpha.contiguous(),
            bank_u8,
            native_codebooks,
            neuron_len,
            assignment_refine_steps,
        )
        nvec = math.ceil(neuron_len / spec.vector_size)
        nsign = math.ceil(neuron_len / 8)
        return NvqJscTensor(
            shape=(out, neuron_len),
            axis=0,
            neuron_len=neuron_len,
            neuron_scale=neuron_scale.cpu().numpy().astype(
                np.float32, copy=False
            ),
            scale_lut=alpha_np,
            bank_for_state=bank_np,
            state=state.cpu().numpy().astype(np.uint8, copy=False),
            indices=(
                indices.reshape(out, ng * _GROUP_SIZE // spec.vector_size)[
                    :, :nvec
                ]
                .cpu()
                .numpy()
                .astype(
                    np.uint8 if spec.index_bits <= 8 else np.uint16,
                    copy=False,
                )
            ),
            signs=signs[:, :nsign].cpu().numpy().astype(
                np.uint8, copy=False
            ),
            codebooks=codebooks_np,
            base_spec=spec,
        )
    if metal_assign is not None:
        bank_peak = codebooks.to(torch.float32).amax((1, 2))
        maximum_basis = (
            alpha * bank_peak[bank_for_state]
        ).amax().clamp_min(1e-20)
        weighted_target = torch.where(
            objective_weight > 0,
            target,
            torch.zeros((), device=target.device, dtype=target.dtype),
        )
        base_neuron_scale = _fp16_round(
            weighted_target[:, :neuron_len].amax(1) / maximum_basis
        ).contiguous()
        anchor_multipliers = (
            (0.875, 1.0, 1.125, 1.25)
            if spec.codebook_entries > 256
            else (1.0,)
        )
        vectors_per_group = _GROUP_SIZE // spec.vector_size
        best_error = torch.full((out,), torch.inf, device=value.device)
        best_neuron_scale = torch.zeros_like(base_neuron_scale)
        best_state_2d = torch.zeros(
            (out, ng), device=value.device, dtype=torch.uint8
        )
        best_indices_3d = torch.zeros(
            (out, ng, vectors_per_group),
            device=value.device,
            dtype=torch.int64,
        )
        for multiplier in anchor_multipliers:
            neuron_scale = _fp16_round(
                base_neuron_scale * float(multiplier)
            ).contiguous()
            state_2d, indices_3d = metal_assign(
                target,
                objective_weight,
                neuron_scale,
                alpha,
                bank_for_state,
                codebooks,
                neuron_len,
            )
            state = state_2d.reshape(-1).to(torch.int64)
            indices = indices_3d.reshape(
                -1, vectors_per_group
            ).to(torch.int64)
            bank = bank_for_state[state]
            # Joint E8 assignment makes repeated full-tensor anchor refits
            # negligible; D4 still benefits from the configured refinement.
            metal_refine_steps = (
                0
                if spec.vector_size == 8
                else assignment_refine_steps
            )
            for _ in range(metal_refine_steps):
                neuron_scale, _ = _refit_scale_tables(
                    xgroup,
                    wgroup,
                    codebooks,
                    state,
                    bank,
                    indices,
                    neuron_scale,
                    alpha,
                    out=out,
                    ng=ng,
                    learned_scale_lut=False,
                )
                state_2d, indices_3d = metal_assign(
                    target,
                    objective_weight,
                    neuron_scale,
                    alpha,
                    bank_for_state,
                    codebooks,
                    neuron_len,
                )
                state = state_2d.reshape(-1).to(torch.int64)
                indices = indices_3d.reshape(
                    -1, vectors_per_group
                ).to(torch.int64)
                bank = bank_for_state[state]
            code = _codes_for_assignment(codebooks, bank, indices)
            effective_scale = (
                neuron_scale.repeat_interleave(ng) * alpha[state]
            )
            row_error = (
                wgroup
                * (
                    effective_scale.unsqueeze(1) * code - xgroup
                ).square()
            ).reshape(out, ng, _GROUP_SIZE).sum((1, 2))
            improve = row_error < best_error
            best_error = torch.where(improve, row_error, best_error)
            best_neuron_scale = torch.where(
                improve, neuron_scale, best_neuron_scale
            )
            best_state_2d = torch.where(
                improve.unsqueeze(1), state_2d, best_state_2d
            )
            best_indices_3d = torch.where(
                improve.reshape(out, 1, 1),
                indices_3d.to(torch.int64),
                best_indices_3d,
            )
        neuron_scale = best_neuron_scale
        state_2d = best_state_2d
        indices_3d = best_indices_3d
        nvec = math.ceil(neuron_len / spec.vector_size)
        nsign = math.ceil(neuron_len / 8)
        return NvqJscTensor(
            shape=(out, neuron_len),
            axis=0,
            neuron_len=neuron_len,
            neuron_scale=neuron_scale.cpu().numpy().astype(
                np.float32, copy=False
            ),
            scale_lut=alpha_np,
            bank_for_state=bank_np,
            state=state_2d.cpu().numpy().astype(np.uint8, copy=False),
            indices=(
                indices_3d.reshape(out, ng * vectors_per_group)[:, :nvec]
                .cpu()
                .numpy()
                .astype(
                    np.uint8 if spec.index_bits <= 8 else np.uint16,
                    copy=False,
                )
            ),
            signs=signs[:, :nsign].cpu().numpy().astype(
                np.uint8, copy=False
            ),
            codebooks=codebooks_np,
            base_spec=spec,
        )
    config = NvqJscConfig(
        banks=codebooks_np.shape[0],
        iterations=0,
        assignment_refine_steps=assignment_refine_steps,
        search_steps=search_steps,
        learned_scale_lut=False,
        codebook_storage="int8",
        group_chunk=group_chunk,
        spec=spec,
    )

    raw_scales = []
    raw_errors = []
    for bank_id in range(codebooks.shape[0]):
        raw_scale, raw_index = _search(
            xgroup, wgroup, codebooks[bank_id], ng, valid_last, config
        )
        raw_code = codebooks[bank_id][raw_index].reshape_as(xgroup).to(torch.float32)
        raw_errors.append(
            (wgroup * (raw_scale.unsqueeze(1) * raw_code - xgroup).square()).sum(1)
        )
        raw_scales.append(raw_scale)
    best_bank = torch.stack(raw_errors, dim=1).argmin(1)
    best_scale = torch.stack(raw_scales, dim=1).gather(
        1, best_bank.unsqueeze(1)
    ).squeeze(1)
    row_max = best_scale.reshape(out, ng).amax(1)
    neuron_scale = _fp16_round(row_max / alpha.max())

    state, bank, indices, _ = _assign_groups(
        xgroup,
        wgroup,
        codebooks,
        neuron_scale,
        alpha,
        bank_for_state,
        out=out,
        ng=ng,
        valid_last=valid_last,
        config=config,
    )
    for _ in range(assignment_refine_steps):
        neuron_scale, _ = _refit_scale_tables(
            xgroup,
            wgroup,
            codebooks,
            state,
            bank,
            indices,
            neuron_scale,
            alpha,
            out=out,
            ng=ng,
            learned_scale_lut=False,
        )
        state, bank, indices, _ = _assign_groups(
            xgroup,
            wgroup,
            codebooks,
            neuron_scale,
            alpha,
            bank_for_state,
            out=out,
            ng=ng,
            valid_last=valid_last,
            config=config,
        )
    if not torch.equal(bank, bank_for_state[state]):
        raise RuntimeError("NVQ-JSC fixed state/bank assignment is inconsistent")

    vectors_per_group = _GROUP_SIZE // spec.vector_size
    nvec = math.ceil(neuron_len / spec.vector_size)
    nsign = math.ceil(neuron_len / 8)
    return NvqJscTensor(
        shape=(out, neuron_len),
        axis=0,
        neuron_len=neuron_len,
        neuron_scale=neuron_scale.cpu().numpy().astype(np.float32, copy=False),
        scale_lut=alpha_np,
        bank_for_state=bank_np,
        state=state.reshape(out, ng).cpu().numpy().astype(np.uint8, copy=False),
        indices=(
            indices.reshape(out, ng * vectors_per_group)[:, :nvec]
            .cpu()
            .numpy()
            .astype(
                np.uint8 if spec.index_bits <= 8 else np.uint16,
                copy=False,
            )
        ),
        signs=signs[:, :nsign].cpu().numpy().astype(np.uint8, copy=False),
        codebooks=codebooks_np,
        base_spec=spec,
    )


@torch.inference_mode()
def train_nvq_jsc(
    weight: torch.Tensor,
    *,
    importance: np.ndarray | torch.Tensor | None = None,
    config: NvqJscConfig = NvqJscConfig(),
    device: str | torch.device = "cuda",
    initial_codebooks: np.ndarray | None = None,
) -> tuple[NvqJscTensor, tuple[NvqJscIteration, ...]]:
    """Train one NVQ-JSC tensor with coordinate descent on weighted SSE."""

    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("NVQ-JSC CUDA training requested without a CUDA device")
    value, out, neuron_len = _prepare_weight(weight, device)
    value, objective_weight, ng = _pad_weight(value, _GROUP_SIZE, importance)
    target, signs = _encode_even_parity_signs(value, objective_weight)
    xgroup = target.reshape(out * ng, _GROUP_SIZE).contiguous()
    wgroup = objective_weight.reshape_as(xgroup).contiguous()
    valid_last = neuron_len - (ng - 1) * _GROUP_SIZE
    raw = (
        initial_raw_codebooks(config)
        if initial_codebooks is None
        else _validate_codebooks(
            initial_codebooks,
            config.banks,
            config.codebook_storage,
            config.spec,
        )
    )
    if initial_codebooks is None and config.codebook_storage == "float32":
        raw = raw.astype(np.float32)
    codebook_dtype = torch.int8 if config.codebook_storage == "int8" else torch.float32
    codebooks = torch.as_tensor(raw, device=value.device, dtype=codebook_dtype).contiguous()
    alpha_cpu, bank_cpu = _initial_state_tables(config)
    alpha = alpha_cpu.to(value.device)
    bank_for_state = bank_cpu.to(value.device)
    metal_fused_assignment = _metal_e8_jsc_assignment_supported(
        codebooks, config
    )

    if metal_fused_assignment:
        candidate_count = min(_METAL_E8_ANCHOR_GROUPS, ng)
        group_peak = xgroup.amax(1).reshape(out, ng)
        candidate_group = group_peak.topk(
            candidate_count, dim=1, sorted=False
        ).indices
        candidate_flat = (
            torch.arange(out, device=value.device).unsqueeze(1) * ng
            + candidate_group
        ).reshape(-1)
        candidate_x = xgroup[candidate_flat].contiguous()
        candidate_w = wgroup[candidate_flat].contiguous()
        raw_scales = []
        raw_errors = []
        for bank_id in range(config.banks):
            raw_scale, raw_index = _search(
                candidate_x,
                candidate_w,
                codebooks[bank_id],
                candidate_count,
                _GROUP_SIZE,
                config,
            )
            raw_code = (
                codebooks[bank_id][raw_index]
                .reshape_as(candidate_x)
                .to(torch.float32)
            )
            raw_error = (
                candidate_w
                * (raw_scale.unsqueeze(1) * raw_code - candidate_x).square()
            ).sum(1)
            raw_scales.append(raw_scale)
            raw_errors.append(raw_error)
        stacked_error = torch.stack(raw_errors, dim=1)
        initial_bank = stacked_error.argmin(1)
        stacked_scale = torch.stack(raw_scales, dim=1)
        initial_scale = stacked_scale.gather(
            1, initial_bank.unsqueeze(1)
        ).squeeze(1)
        row_max = initial_scale.reshape(out, candidate_count).amax(1)
        alpha_max = alpha.max()
        neuron_scale = _fp16_round(
            torch.where(row_max > 0, row_max / alpha_max, row_max)
        ).contiguous()
    elif _native_e8_jsc_assignment_supported(codebooks, config):
        raw_scales = []
        raw_errors = []
        native_scales, native_indices = _native_e8_search_banks(
            xgroup,
            wgroup,
            codebooks,
            ng=ng,
            valid_last=valid_last,
            search_steps=config.search_steps,
        )
        for bank_id in range(config.banks):
            raw_scale = native_scales[:, bank_id]
            raw_index = native_indices[:, bank_id]
            raw_code = (
                codebooks[bank_id][raw_index]
                .reshape_as(xgroup)
                .to(torch.float32)
            )
            raw_error = (
                wgroup
                * (raw_scale.unsqueeze(1) * raw_code - xgroup).square()
            ).sum(1)
            raw_scales.append(raw_scale)
            raw_errors.append(raw_error)
        stacked_error = torch.stack(raw_errors, dim=1)
        initial_bank = stacked_error.argmin(1)
        stacked_scale = torch.stack(raw_scales, dim=1)
        initial_scale = stacked_scale.gather(
            1, initial_bank.unsqueeze(1)
        ).squeeze(1)
        row_max = initial_scale.reshape(out, ng).amax(1)
        alpha_max = alpha.max()
        neuron_scale = _fp16_round(
            torch.where(row_max > 0, row_max / alpha_max, row_max)
        )
    else:
        raw_scales = []
        raw_errors = []
        for bank_id in range(config.banks):
            raw_scale, raw_index = _search(
                xgroup,
                wgroup,
                codebooks[bank_id],
                ng,
                valid_last,
                config,
            )
            raw_code = (
                codebooks[bank_id][raw_index]
                .reshape_as(xgroup)
                .to(torch.float32)
            )
            raw_error = (
                wgroup
                * (raw_scale.unsqueeze(1) * raw_code - xgroup).square()
            ).sum(1)
            raw_scales.append(raw_scale)
            raw_errors.append(raw_error)
        stacked_error = torch.stack(raw_errors, dim=1)
        initial_bank = stacked_error.argmin(1)
        stacked_scale = torch.stack(raw_scales, dim=1)
        initial_scale = stacked_scale.gather(
            1, initial_bank.unsqueeze(1)
        ).squeeze(1)
        row_max = initial_scale.reshape(out, ng).amax(1)
        alpha_max = alpha.max()
        neuron_scale = _fp16_round(
            torch.where(row_max > 0, row_max / alpha_max, row_max)
        )

    history: list[NvqJscIteration] = []
    best: tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    signal = float((wgroup * xgroup.square()).sum().item())

    iteration_limit = config.iterations
    if metal_fused_assignment and config.iterations >= 4:
        iteration_limit += _METAL_E8_EXTRA_ITERATIONS
    for iteration in range(iteration_limit + 1):
        state, bank, indices, _ = _assign_groups(
            xgroup,
            wgroup,
            codebooks,
            neuron_scale,
            alpha,
            bank_for_state,
            out=out,
            ng=ng,
            valid_last=valid_last,
            config=config,
        )
        refine_steps = (
            0 if metal_fused_assignment else config.assignment_refine_steps
        )
        for _ in range(refine_steps):
            neuron_scale, alpha = _refit_scale_tables(
                xgroup,
                wgroup,
                codebooks,
                state,
                bank,
                indices,
                neuron_scale,
                alpha,
                out=out,
                ng=ng,
                learned_scale_lut=config.learned_scale_lut,
            )
            state, bank, indices, _ = _assign_groups(
                xgroup,
                wgroup,
                codebooks,
                neuron_scale,
                alpha,
                bank_for_state,
                out=out,
                ng=ng,
                valid_last=valid_last,
                config=config,
            )
        error = float(
            _assignment_error(
                xgroup,
                wgroup,
                codebooks,
                state,
                bank,
                indices,
                neuron_scale,
                alpha,
                ng=ng,
            ).item()
        )
        used_codes = tuple(
            int(torch.unique(indices[bank == bank_id]).numel())
            if bool((bank == bank_id).any())
            else 0
            for bank_id in range(config.banks)
        )
        history.append(
            NvqJscIteration(
                iteration=iteration,
                weighted_sse=error,
                weighted_nmse_percent=100.0 * error / signal if signal else 0.0,
                used_states=int(torch.unique(state).numel()),
                used_banks=int(torch.unique(bank).numel()),
                used_codes=used_codes,
            )
        )
        if best is None or error < best[0]:
            best = (
                error,
                neuron_scale.clone(),
                alpha.clone(),
                state.clone(),
                bank.clone(),
                indices.clone(),
            )
            best_codebooks = codebooks.clone()
        if iteration == iteration_limit:
            break
        codebooks = _update_codebooks(
            xgroup,
            wgroup,
            codebooks,
            state,
            bank,
            indices,
            neuron_scale,
            alpha,
            ng=ng,
            storage=config.codebook_storage,
        )

    assert best is not None
    _, neuron_scale, alpha, state, bank, indices = best
    codebooks = best_codebooks
    if not torch.equal(bank, bank_for_state[state]):
        raise RuntimeError("NVQ-JSC state/bank assignment is inconsistent")
    vectors_per_group = _GROUP_SIZE // config.spec.vector_size
    nvec = math.ceil(neuron_len / config.spec.vector_size)
    nsign = math.ceil(neuron_len / 8)
    tensor = NvqJscTensor(
        shape=(out, neuron_len),
        axis=0,
        neuron_len=neuron_len,
        neuron_scale=neuron_scale.cpu().numpy().astype(np.float32, copy=False),
        scale_lut=alpha.cpu().numpy().astype(np.float32, copy=False),
        bank_for_state=bank_for_state.cpu().numpy().astype(np.uint8, copy=False),
        state=state.reshape(out, ng).cpu().numpy().astype(np.uint8, copy=False),
        indices=(
            indices.reshape(out, ng * vectors_per_group)[:, :nvec]
            .cpu()
            .numpy()
            .astype(
                np.uint8 if config.spec.index_bits <= 8 else np.uint16,
                copy=False,
            )
        ),
        signs=signs[:, :nsign].cpu().numpy().astype(np.uint8, copy=False),
        codebooks=codebooks.cpu().numpy(),
        base_spec=config.spec,
    )
    return tensor, tuple(history)


def jsc_tables_from_tensor(tensor: NvqJscTensor) -> NvqJscTables:
    return NvqJscTables(
        scale_lut=np.asarray(tensor.scale_lut, dtype=np.float32).copy(),
        bank_for_state=np.asarray(tensor.bank_for_state, dtype=np.uint8).copy(),
        codebooks=np.asarray(tensor.codebooks, dtype=np.int8).copy(),
        spec=tensor.spec,
    )


def dequantize_nvq_jsc(tensor: NvqJscTensor) -> np.ndarray:
    out, neuron_len = tensor.shape
    ng = math.ceil(neuron_len / _GROUP_SIZE)
    vectors_per_group = _GROUP_SIZE // tensor.spec.vector_size
    nvec_padded = ng * vectors_per_group
    indices = np.zeros(
        (out, nvec_padded),
        dtype=np.uint8 if tensor.spec.index_bits <= 8 else np.uint16,
    )
    indices[:, : tensor.indices.shape[1]] = tensor.indices
    group_bank = tensor.bank_for_state[tensor.state]
    vector_bank = np.repeat(group_bank, vectors_per_group, axis=1)
    magnitude = tensor.codebooks[vector_bank, indices].reshape(out, -1)[:, :neuron_len]

    masks = np.asarray(tensor.signs, dtype=np.uint8)
    lower = ((masks[..., None] >> np.arange(7, dtype=np.uint8)) & 1).astype(np.uint8)
    last = (lower.sum(axis=-1, keepdims=True) & 1).astype(np.uint8)
    negative = np.concatenate([lower, last], axis=-1)
    sign = np.where(negative != 0, -1.0, 1.0).astype(np.float32)
    sign = sign.reshape(out, -1)[:, :neuron_len]

    group_scale = (
        tensor.neuron_scale[:, None]
        * tensor.scale_lut[tensor.state].astype(np.float32)
    )
    scale = np.repeat(group_scale, _GROUP_SIZE, axis=1)[:, :neuron_len]
    return magnitude.astype(np.float32) * sign * scale


__all__ = [
    "NvqJscConfig",
    "NvqJscIteration",
    "NvqJscTables",
    "NvqJscTensor",
    "dequantize_nvq_jsc",
    "initial_jsc_tables",
    "initial_raw_codebooks",
    "jsc_tables_from_tensor",
    "quantize_nvq_jsc_fixed",
    "train_nvq_jsc",
]
