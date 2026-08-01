"""Activation-aware second-order refinements for NVQ2 experiments."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import torch

from mfq.formats.nvq import NvqJscTensor, NvqTensor, codebook_for
from mfq.quantize.nvq_jsc import dequantize_nvq_jsc
from mfq.quantize.nvq_quant import dequantize as dequantize_nvq


@dataclass(frozen=True)
class LinearOutputMetrics:
    nmse_percent: float
    snr_db: float
    row_nmse_percent: np.ndarray


@dataclass(frozen=True)
class Block24Iteration:
    iteration: int
    block_objective: float
    train_output_nmse_percent: float
    changed_indices_percent: float
    changed_levels_percent: float


@dataclass(frozen=True)
class JscBlock24Iteration:
    iteration: int
    block_objective: float
    train_output_nmse_percent: float
    changed_indices_percent: float
    changed_states_percent: float


def _as_cuda_matrix(
    value: np.ndarray | torch.Tensor,
    device: str | torch.device,
) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=torch.float32).contiguous()


@torch.inference_mode()
def linear_output_metrics(
    weight: np.ndarray | torch.Tensor,
    reconstruction: np.ndarray | torch.Tensor,
    activations: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device = "cuda",
    token_chunk: int = 512,
) -> LinearOutputMetrics:
    """Measure linear output error without retaining all token outputs."""

    if token_chunk <= 0:
        raise ValueError("token_chunk must be positive")
    reference = _as_cuda_matrix(weight, device)
    quantized = _as_cuda_matrix(reconstruction, device)
    inputs = _as_cuda_matrix(activations, device)
    if reference.shape != quantized.shape:
        raise ValueError("weight and reconstruction shapes differ")
    if inputs.dim() != 2 or inputs.shape[1] != reference.shape[1]:
        raise ValueError("activation width does not match weight input width")

    signal = torch.zeros(reference.shape[0], device=reference.device)
    error = torch.zeros_like(signal)
    for start in range(0, inputs.shape[0], token_chunk):
        x = inputs[start : start + token_chunk]
        y = x @ reference.T
        z = x @ quantized.T
        signal += y.square().sum(0)
        error += (y - z).square().sum(0)
    row_nmse = torch.where(signal > 0, 100.0 * error / signal, torch.zeros_like(signal))
    total_signal = float(signal.sum().item())
    total_error = float(error.sum().item())
    return LinearOutputMetrics(
        nmse_percent=100.0 * total_error / total_signal if total_signal else 0.0,
        snr_db=(10.0 * math.log10(total_signal / total_error) if total_error else math.inf),
        row_nmse_percent=row_nmse.cpu().numpy(),
    )


@torch.inference_mode()
def activation_regressed_gain(
    weight: np.ndarray | torch.Tensor,
    reconstruction: np.ndarray | torch.Tensor,
    activations: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device = "cuda",
    token_chunk: int = 512,
) -> np.ndarray:
    """Fit one non-negative output gain per row on real linear inputs."""

    if token_chunk <= 0:
        raise ValueError("token_chunk must be positive")
    reference = _as_cuda_matrix(weight, device)
    quantized = _as_cuda_matrix(reconstruction, device)
    inputs = _as_cuda_matrix(activations, device)
    if reference.shape != quantized.shape:
        raise ValueError("weight and reconstruction shapes differ")
    if inputs.dim() != 2 or inputs.shape[1] != reference.shape[1]:
        raise ValueError("activation width does not match weight input width")

    numerator = torch.zeros(reference.shape[0], device=reference.device)
    denominator = torch.zeros_like(numerator)
    for start in range(0, inputs.shape[0], token_chunk):
        x = inputs[start : start + token_chunk]
        y = x @ reference.T
        z = x @ quantized.T
        numerator += (y * z).sum(0)
        denominator += z.square().sum(0)
    gain = torch.where(
        denominator > 0,
        numerator / denominator,
        torch.ones_like(denominator),
    ).clamp_min(0)
    return gain.cpu().numpy().astype(np.float32, copy=False)


def diagonal_regressed_gain(
    weight: np.ndarray | torch.Tensor,
    reconstruction: np.ndarray | torch.Tensor,
    importance: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Fit per-row output gains under a diagonal activation covariance."""

    reference = np.asarray(weight, dtype=np.float32)
    quantized = np.asarray(reconstruction, dtype=np.float32)
    diagonal = np.asarray(importance, dtype=np.float32)
    if reference.shape != quantized.shape or reference.ndim != 2:
        raise ValueError("weight and reconstruction must be matching matrices")
    if diagonal.ndim == 1:
        if diagonal.shape[0] != reference.shape[1]:
            raise ValueError("importance width does not match weight input width")
        diagonal = np.broadcast_to(diagonal, reference.shape)
    elif diagonal.shape != reference.shape:
        raise ValueError("importance must be [in] or [out,in]")
    if not np.isfinite(diagonal).all() or np.any(diagonal < 0):
        raise ValueError("importance must be finite and non-negative")
    numerator = np.einsum(
        "ij,ij,ij->i", diagonal, reference, quantized, dtype=np.float64
    )
    denominator = np.einsum(
        "ij,ij,ij->i", diagonal, quantized, quantized, dtype=np.float64
    )
    gain = np.ones(reference.shape[0], dtype=np.float64)
    valid = denominator > 0
    gain[valid] = np.maximum(numerator[valid] / denominator[valid], 0.0)
    return gain.astype(np.float32)


def apply_neuron_gain(
    tensor: NvqTensor | NvqJscTensor,
    gain: np.ndarray,
) -> NvqTensor | NvqJscTensor:
    """Absorb per-row gain into the existing FP16 neuron anchor."""

    value = np.asarray(gain, dtype=np.float32).reshape(-1)
    if value.shape != tensor.neuron_scale.shape:
        raise ValueError(
            f"gain shape {value.shape} does not match anchors {tensor.neuron_scale.shape}"
        )
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError("gain must be finite and non-negative")
    anchor = (
        np.asarray(tensor.neuron_scale, dtype=np.float32) * value
    ).astype(np.float16).astype(np.float32)
    return dataclasses.replace(tensor, neuron_scale=anchor)


def median_row_norm_gain(
    weight: np.ndarray,
    reconstruction: np.ndarray,
) -> float:
    """Return the production-style median row L2 correction."""

    reference_energy = np.einsum("ij,ij->i", weight, weight, dtype=np.float64)
    quantized_energy = np.einsum(
        "ij,ij->i", reconstruction, reconstruction, dtype=np.float64
    )
    valid = (reference_energy > 0) & (quantized_energy > 0)
    correction = np.ones(weight.shape[0], dtype=np.float64)
    correction[valid] = np.sqrt(reference_energy[valid] / quantized_energy[valid])
    return float(np.median(correction))


def _decode_signs(tensor: NvqTensor, padded_len: int) -> torch.Tensor:
    masks = torch.as_tensor(tensor.signs, dtype=torch.int64)
    shifts = torch.arange(7, dtype=torch.int64)
    lower = (masks.unsqueeze(-1) >> shifts) & 1
    last = lower.sum(-1, keepdim=True) & 1
    negative = torch.cat((lower, last), dim=-1).reshape(masks.shape[0], -1)
    sign = torch.where(negative != 0, -torch.ones_like(negative), torch.ones_like(negative))
    if sign.shape[1] < padded_len:
        sign = torch.nn.functional.pad(sign, (0, padded_len - sign.shape[1]), value=1)
    return sign[:, :padded_len].to(torch.float32)


def _build_block_hessian(activations: torch.Tensor, padded_len: int) -> torch.Tensor:
    if activations.shape[1] < padded_len:
        activations = torch.nn.functional.pad(
            activations,
            (0, padded_len - activations.shape[1]),
        )
    blocks = activations.reshape(activations.shape[0], padded_len // 24, 24)
    return torch.einsum("tgi,tgj->gij", blocks, blocks) / float(activations.shape[0])


def _jsc_group_basis(
    codebooks: torch.Tensor,
    bank_for_state: torch.Tensor,
    alpha: torch.Tensor,
    state: torch.Tensor,
    indices: torch.Tensor,
    sign: torch.Tensor,
) -> torch.Tensor:
    bank = bank_for_state[state]
    code = codebooks[bank[:, None], indices].reshape(-1, 24)
    return alpha[state, None] * code * sign


def _refine_jsc_group_chunk(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    sign: torch.Tensor,
    anchor: torch.Tensor,
    state: torch.Tensor,
    indices: torch.Tensor,
    codebooks: torch.Tensor,
    alpha: torch.Tensor,
    bank_for_state: torch.Tensor,
    sweeps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(weight.shape[0], device=weight.device)
    state_codebooks = codebooks[bank_for_state]
    for _ in range(sweeps):
        basis = _jsc_group_basis(
            codebooks, bank_for_state, alpha, state, indices, sign
        )
        quantized = anchor[:, None] * basis
        residual = quantized - weight

        for vector in range(3):
            start = vector * 8
            stop = start + 8
            bank = bank_for_state[state]
            signed_codebook = (
                codebooks[bank] * sign[:, None, start:stop]
            )
            scale = anchor * alpha[state]
            candidate = scale[:, None, None] * signed_codebook
            delta = candidate - quantized[:, None, start:stop]
            h_error = torch.einsum("gij,gj->gi", hessian, residual)
            linear = 2.0 * torch.einsum(
                "gci,gi->gc", delta, h_error[:, start:stop]
            )
            local_hessian = hessian[:, start:stop, start:stop]
            quadratic = torch.einsum(
                "gci,gij,gcj->gc", delta, local_hessian, delta
            )
            selected = (linear + quadratic).argmin(1)
            chosen = candidate[rows, selected]
            quantized[:, start:stop] = chosen
            residual[:, start:stop] = chosen - weight[:, start:stop]
            indices[:, vector] = selected

        candidate_code = (
            state_codebooks[:, indices]
            .permute(1, 0, 2, 3)
            .reshape(weight.shape[0], 16, 24)
        )
        candidate_basis = candidate_code * sign[:, None, :]
        candidate = (
            anchor[:, None, None]
            * alpha[None, :, None]
            * candidate_basis
        )
        candidate_residual = candidate - weight[:, None, :]
        error = torch.einsum(
            "gsi,gij,gsj->gs", candidate_residual, hessian, candidate_residual
        )
        state = error.argmin(1)
    return state, indices


@torch.inference_mode()
def refine_nvq2j_block24(
    weight: np.ndarray | torch.Tensor,
    tensor: NvqJscTensor,
    activations: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device = "cuda",
    outer_iterations: int = 2,
    coordinate_sweeps: int = 2,
    group_chunk: int = 512,
) -> tuple[NvqJscTensor, tuple[JscBlock24Iteration, ...]]:
    """Refine JSC states/indices/anchors under a block-diagonal 24x24 Hessian."""

    if outer_iterations < 1 or coordinate_sweeps < 1 or group_chunk <= 0:
        raise ValueError("refinement iteration and chunk counts must be positive")
    value = _as_cuda_matrix(weight, device)
    inputs = _as_cuda_matrix(activations, device)
    out, neuron_len = value.shape
    if tuple(tensor.shape) != (out, neuron_len) or tensor.axis != 0:
        raise ValueError("NVQ2J tensor shape does not match weight")
    if inputs.dim() != 2 or inputs.shape[1] != neuron_len:
        raise ValueError("activation width does not match weight input width")

    ng = math.ceil(neuron_len / 24)
    padded_len = ng * 24
    if neuron_len < padded_len:
        value = torch.nn.functional.pad(value, (0, padded_len - neuron_len))
    hessian = _build_block_hessian(inputs, padded_len)
    sign = _decode_signs(tensor, padded_len).to(device).reshape(out, ng, 24)
    codebooks = torch.as_tensor(
        tensor.codebooks, device=device, dtype=torch.float32
    )
    alpha = torch.as_tensor(tensor.scale_lut, device=device, dtype=torch.float32)
    bank_for_state = torch.as_tensor(
        tensor.bank_for_state, device=device, dtype=torch.int64
    )
    state = torch.as_tensor(tensor.state, device=device, dtype=torch.int64).clone()
    nvec_padded = ng * 3
    indices = torch.zeros((out, nvec_padded), device=device, dtype=torch.int64)
    indices[:, : tensor.indices.shape[1]] = torch.as_tensor(
        tensor.indices, device=device, dtype=torch.int64
    )
    indices = indices.reshape(out, ng, 3)
    anchor = torch.as_tensor(
        tensor.neuron_scale, device=device, dtype=torch.float32
    ).clone()
    original_state = state.clone()
    original_indices = indices.clone()
    weight_group = value.reshape(out, ng, 24)
    history: list[JscBlock24Iteration] = []

    for iteration in range(outer_iterations):
        flat_weight = weight_group.reshape(out * ng, 24)
        flat_sign = sign.reshape(out * ng, 24)
        flat_state = state.reshape(-1)
        flat_indices = indices.reshape(out * ng, 3)
        flat_anchor = anchor.repeat_interleave(ng)
        for start in range(0, out * ng, group_chunk):
            stop = min(start + group_chunk, out * ng)
            block_ids = torch.arange(start, stop, device=value.device) % ng
            new_state, new_indices = _refine_jsc_group_chunk(
                flat_weight[start:stop],
                hessian[block_ids],
                flat_sign[start:stop],
                flat_anchor[start:stop],
                flat_state[start:stop],
                flat_indices[start:stop],
                codebooks,
                alpha,
                bank_for_state,
                coordinate_sweeps,
            )
            flat_state[start:stop] = new_state
            flat_indices[start:stop] = new_indices
        state = flat_state.reshape(out, ng)
        indices = flat_indices.reshape(out, ng, 3)

        basis = _jsc_group_basis(
            codebooks,
            bank_for_state,
            alpha,
            state.reshape(-1),
            indices.reshape(out * ng, 3),
            sign.reshape(out * ng, 24),
        ).reshape(out, ng, 24)
        numerator = torch.einsum("ogi,gij,ogj->o", weight_group, hessian, basis)
        denominator = torch.einsum("ogi,gij,ogj->o", basis, hessian, basis)
        fitted_anchor = torch.where(
            denominator > 0, numerator / denominator, anchor
        ).clamp_min(0).to(torch.float16).to(torch.float32)
        old_residual = anchor[:, None, None] * basis - weight_group
        new_residual = fitted_anchor[:, None, None] * basis - weight_group
        old_error = torch.einsum("ogi,gij,ogj->o", old_residual, hessian, old_residual)
        new_error = torch.einsum("ogi,gij,ogj->o", new_residual, hessian, new_residual)
        anchor = torch.where(new_error <= old_error, fitted_anchor, anchor)

        quantized = anchor[:, None, None] * basis
        residual = quantized - weight_group
        block_objective = float(
            torch.einsum("ogi,gij,ogj->", residual, hessian, residual).item()
        )
        reconstruction = quantized.reshape(out, padded_len)[:, :neuron_len]
        train_metrics = linear_output_metrics(
            value[:, :neuron_len], reconstruction, inputs, device=device
        )
        history.append(
            JscBlock24Iteration(
                iteration=iteration,
                block_objective=block_objective,
                train_output_nmse_percent=train_metrics.nmse_percent,
                changed_indices_percent=float(
                    100.0 * (indices != original_indices).float().mean().item()
                ),
                changed_states_percent=float(
                    100.0 * (state != original_state).float().mean().item()
                ),
            )
        )

    nvec = math.ceil(neuron_len / 8)
    result = dataclasses.replace(
        tensor,
        neuron_scale=anchor.cpu().numpy().astype(np.float32, copy=False),
        state=state.cpu().numpy().astype(np.uint8, copy=False),
        indices=(
            indices.reshape(out, nvec_padded)[:, :nvec]
            .cpu()
            .numpy()
            .astype(np.uint8, copy=False)
        ),
    )
    return result, tuple(history)


def _codes(
    codebook: torch.Tensor,
    indices: torch.Tensor,
    sign: torch.Tensor,
) -> torch.Tensor:
    magnitude = codebook[indices].reshape(indices.shape[0], -1)
    return magnitude * sign


def _choose_levels(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    basis: torch.Tensor,
    anchor: torch.Tensor,
) -> torch.Tensor:
    levels = torch.arange(16, device=weight.device, dtype=torch.float32)
    candidate = anchor[:, None, None] * levels[None, :, None] * basis[:, None, :]
    residual = candidate - weight[:, None, :]
    error = torch.einsum("gli,gij,glj->gl", residual, hessian, residual)
    return error.argmin(1).to(torch.uint8)


def _refine_group_chunk(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    sign: torch.Tensor,
    anchor: torch.Tensor,
    level: torch.Tensor,
    indices: torch.Tensor,
    codebook: torch.Tensor,
    sweeps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(weight.shape[0], device=weight.device)
    for _ in range(sweeps):
        basis = _codes(codebook, indices, sign)
        level = _choose_levels(weight, hessian, basis, anchor)
        scale = anchor * level.to(torch.float32)
        quantized = scale.unsqueeze(1) * basis
        residual = quantized - weight

        for vector in range(3):
            start = vector * 8
            stop = start + 8
            signed_codebook = sign[:, None, start:stop] * codebook.unsqueeze(0)
            candidate = scale[:, None, None] * signed_codebook
            delta = candidate - quantized[:, None, start:stop]
            h_error = torch.einsum("gij,gj->gi", hessian, residual)
            linear = 2.0 * torch.einsum(
                "gci,gi->gc", delta, h_error[:, start:stop]
            )
            local_hessian = hessian[:, start:stop, start:stop]
            quadratic = torch.einsum(
                "gci,gij,gcj->gc",
                delta,
                local_hessian,
                delta,
            )
            selected = (linear + quadratic).argmin(1)
            chosen = candidate[rows, selected]
            quantized[:, start:stop] = chosen
            residual[:, start:stop] = chosen - weight[:, start:stop]
            indices[:, vector] = selected

        basis = _codes(codebook, indices, sign)
        level = _choose_levels(weight, hessian, basis, anchor)
    return level, indices


@torch.inference_mode()
def refine_nvq2_block24(
    weight: np.ndarray | torch.Tensor,
    tensor: NvqTensor,
    activations: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device = "cuda",
    outer_iterations: int = 2,
    coordinate_sweeps: int = 2,
    group_chunk: int = 512,
) -> tuple[NvqTensor, tuple[Block24Iteration, ...]]:
    """Refine NVQ2 levels/indices/anchors under a block-diagonal 24x24 Hessian."""

    if tensor.spec.groupsize != 24 or tensor.spec.vector_size != 8:
        raise ValueError("block24 refinement requires NVQ gs24 with vector size 8")
    if outer_iterations < 1 or coordinate_sweeps < 1 or group_chunk <= 0:
        raise ValueError("refinement iteration and chunk counts must be positive")
    value = _as_cuda_matrix(weight, device)
    inputs = _as_cuda_matrix(activations, device)
    out, neuron_len = value.shape
    if tuple(tensor.shape) != (out, neuron_len):
        raise ValueError("NVQ tensor shape does not match weight")
    if inputs.shape[1] != neuron_len:
        raise ValueError("activation width does not match weight input width")

    ng = math.ceil(neuron_len / 24)
    padded_len = ng * 24
    if neuron_len < padded_len:
        value = torch.nn.functional.pad(value, (0, padded_len - neuron_len))
    hessian = _build_block_hessian(inputs, padded_len)
    sign = _decode_signs(tensor, padded_len).to(device)
    codebook_np = codebook_for(tensor.spec) if tensor.codebook is None else tensor.codebook
    codebook = torch.as_tensor(codebook_np, device=device, dtype=torch.float32)
    nvec_padded = ng * 3
    indices = torch.zeros((out, nvec_padded), device=device, dtype=torch.int64)
    indices[:, : tensor.indices.shape[1]] = torch.as_tensor(
        tensor.indices,
        device=device,
        dtype=torch.int64,
    )
    indices = indices.reshape(out, ng, 3)
    level = torch.as_tensor(tensor.sub_scale, device=device, dtype=torch.uint8).clone()
    anchor = torch.as_tensor(
        tensor.neuron_scale,
        device=device,
        dtype=torch.float32,
    ).clone()
    original_indices = indices.clone()
    original_level = level.clone()
    weight_group = value.reshape(out, ng, 24)
    sign_group = sign.reshape(out, ng, 24)
    history: list[Block24Iteration] = []

    for iteration in range(outer_iterations):
        flat_weight = weight_group.reshape(out * ng, 24)
        flat_sign = sign_group.reshape(out * ng, 24)
        flat_level = level.reshape(-1)
        flat_indices = indices.reshape(out * ng, 3)
        flat_anchor = anchor.repeat_interleave(ng)
        for start in range(0, out * ng, group_chunk):
            stop = min(start + group_chunk, out * ng)
            block_ids = torch.arange(start, stop, device=value.device) % ng
            new_level, new_indices = _refine_group_chunk(
                flat_weight[start:stop],
                hessian[block_ids],
                flat_sign[start:stop],
                flat_anchor[start:stop],
                flat_level[start:stop],
                flat_indices[start:stop],
                codebook,
                coordinate_sweeps,
            )
            flat_level[start:stop] = new_level
            flat_indices[start:stop] = new_indices
        level = flat_level.reshape(out, ng)
        indices = flat_indices.reshape(out, ng, 3)

        basis = (
            level.to(torch.float32).unsqueeze(-1)
            * _codes(
                codebook,
                indices.reshape(out * ng, 3),
                sign_group.reshape(out * ng, 24),
            ).reshape(out, ng, 24)
        )
        numerator = torch.einsum("ogi,gij,ogj->o", weight_group, hessian, basis)
        denominator = torch.einsum("ogi,gij,ogj->o", basis, hessian, basis)
        anchor = torch.where(
            denominator > 0,
            numerator / denominator,
            anchor,
        ).clamp_min(0)
        anchor = anchor.to(torch.float16).to(torch.float32)

        quantized = anchor[:, None, None] * basis
        residual = quantized - weight_group
        block_objective = float(
            torch.einsum("ogi,gij,ogj->", residual, hessian, residual).item()
        )
        reconstructed = quantized.reshape(out, padded_len)[:, :neuron_len]
        train_metrics = linear_output_metrics(
            value[:, :neuron_len],
            reconstructed,
            inputs,
            device=device,
        )
        history.append(
            Block24Iteration(
                iteration=iteration,
                block_objective=block_objective,
                train_output_nmse_percent=train_metrics.nmse_percent,
                changed_indices_percent=float(
                    100.0 * (indices != original_indices).float().mean().item()
                ),
                changed_levels_percent=float(
                    100.0 * (level != original_level).float().mean().item()
                ),
            )
        )

    nvec = math.ceil(neuron_len / 8)
    result = dataclasses.replace(
        tensor,
        neuron_scale=anchor.cpu().numpy().astype(np.float32, copy=False),
        sub_scale=level.cpu().numpy().astype(np.uint8, copy=False),
        indices=(
            indices.reshape(out, nvec_padded)[:, :nvec]
            .cpu()
            .numpy()
            .astype(np.uint8, copy=False)
        ),
    )
    return result, tuple(history)


def reconstruct_with_gain(
    tensor: NvqTensor | NvqJscTensor,
    gain: np.ndarray,
) -> tuple[NvqTensor | NvqJscTensor, np.ndarray]:
    calibrated = apply_neuron_gain(tensor, gain)
    reconstruction = (
        dequantize_nvq_jsc(calibrated)
        if isinstance(calibrated, NvqJscTensor)
        else dequantize_nvq(calibrated)
    )
    return calibrated, reconstruction
