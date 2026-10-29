"""Fixed-book CUDA Trellis fitting with optional diagonal imatrix weighting.

``quantize(weight, bits, imatrix=...)`` accepts a scalar width or one width
per output row. Scale-factor trials select each row by the actual FP16
reconstruction error. Offline grouping by width never changes row order.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from mfq.formats.tq import RELATIVE, TQTensor, fixed_codebook
from mfq.quantize.cuda.tq_encoder import encode
from mfq.quantize.cuda.tq_fitter import fit_columns, fit_rows, pack, prepare_columns


@torch.inference_mode()
def quantize(
    weight: torch.Tensor,
    bits: int | np.ndarray,
    *,
    imatrix: torch.Tensor | None = None,
    groupsize: int = 512,
    scale_factors: tuple[float, ...] | None = None,
    search: str = "fast",
    state_bits: int | np.ndarray = 16,
    column_scaling: bool = True,
    repair_scales: bool = True,
) -> TQTensor:
    """Fit a CUDA 2D weight tensor; return a standalone host payload.

    Default fast shared-column fitting covers the complete source range.
    Sparse scale repair accepts only improved FP16 weighted reconstruction.
    Adaptive subgroup fitting screens the retained scale grid on sampled rows.
    ``column_scaling=False, search="full"`` retains the FP32 all-row scale search.
    ``search="fast"`` uses one sampled scale candidate without re-encoding.
    All paths retain the complete 16-bit state space and RTRELL05 payloads.
    ``column_scaling=True`` shares one FP16 input scale vector across all
    row precisions; it is prepared independently of the selected precisions.
    Logical widths are padded to whole rings and subgroups; retain the
    original width when constructing the CUDA ``Projection``.
    """
    if not weight.is_cuda or weight.ndim != 2 or min(weight.shape) <= 0:
        raise ValueError("Expected a nonempty CUDA 2D weight tensor")
    if groupsize < 16 or groupsize > 2**20 or groupsize & (groupsize - 1):
        raise ValueError("groupsize must be a power of two in [16, 2**20]")
    if search not in ("adaptive", "full", "fast"):
        raise ValueError("search must be adaptive, full or fast")
    if search == "fast" and not column_scaling:
        raise ValueError("fast search requires column_scaling=True")
    rows, logical = weight.shape
    widths = np.broadcast_to(np.asarray(bits), (rows,))
    if not np.isin(widths, np.arange(1, 9)).all():
        raise ValueError("Each row must select 1..8 bits")
    state_widths = np.broadcast_to(np.asarray(state_bits), (rows,))
    if not np.all(state_widths == 16):
        raise ValueError("This backend requires complete 16-bit states")
    shorter = state_widths != 16
    if shorter.any() and (search == "full" or scale_factors is not None or (widths[shorter] > 4).any()):
        raise ValueError("Short states require adaptive/fast 1..4-bit fitting and the default scale grid")
    if column_scaling and (search == "full" or scale_factors is not None):
        raise ValueError("Column fitting requires adaptive/fast 1..8-bit fitting and the default scale grid")
    if scale_factors is not None and (
        not scale_factors or any(not math.isfinite(f) or f <= 0 for f in scale_factors)
    ):
        raise ValueError("Scale factors must be finite and positive")
    alignment = 256 if column_scaling else max(256, groupsize)
    width = math.ceil(logical / alignment) * alignment
    with torch.cuda.device(weight.device):
        w = torch.nn.functional.pad(weight.float(), (0, width - logical))
        if not torch.isfinite(w).all():
            raise ValueError("Weights must be finite")
        h = (
            torch.ones(logical, device=w.device)
            if imatrix is None
            else imatrix.to(w.device).float()
        )
        if h.shape != (logical,) or not torch.isfinite(h).all() or (h < 0).any():
            raise ValueError("imatrix must be a finite nonnegative input-channel vector")
        if not (h > 0).any():
            raise ValueError("imatrix must contain a positive weight")
        h = torch.nn.functional.pad(h, (0, width - logical))
        if not column_scaling:
            lut = torch.tensor(fixed_codebook(), device=w.device)
            relative = torch.tensor(RELATIVE, device=w.device).float()
            code_rms = lut.float().square().mean().sqrt()
        row_payloads = [None] * rows
        descriptors = np.zeros(rows, dtype=np.uint64)
        selectors = np.zeros((rows, width // groupsize), dtype=np.uint8)
        column_scales = prepare_columns(w) if column_scaling else None
        profiles = np.stack((widths, state_widths), axis=-1)
        unique_profiles = np.unique(profiles, axis=0)
        for k, state_width in unique_profiles:
            k, state_width = int(k), int(state_width)
            host_ids = np.flatnonzero((widths == k) & (state_widths == state_width))
            ids = torch.tensor(host_ids, device=w.device)
            target = w if len(host_ids) == rows else w[ids]
            if column_scaling:
                best_states, best_anchor = fit_columns(
                    target, k, h, column_scales, state_bits=state_width, fast=search == "fast", repair_scales=repair_scales
                )
                codes = None
            elif search in ("adaptive", "fast") and k <= 4 and scale_factors is None:
                best_states, best_anchor, codes = fit_rows(
                    target, k, h, groupsize, state_bits=state_width, fast=search == "fast"
                )
            else:
                anchor = (target.square().mean(-1).sqrt() / code_rms).clamp_min(2**-24).half()
                wanted = target.reshape(len(ids), -1, groupsize).square().mean(-1).sqrt()
                wanted = wanted / code_rms / anchor[:, None].float()
                codes = (
                    (wanted.clamp_min(2**-24).log2()[..., None] - relative.log2()).abs().argmin(-1)
                )
                factors = scale_factors
                if factors is None:
                    factors = (
                        (0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3)
                        if k == 1
                        else (1.0, 0.9, 1.05, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0)
                    )
                best_loss = torch.full((len(ids),), float("inf"), device=w.device)
                best_anchor = torch.empty_like(anchor)
                best_states = torch.empty((len(ids), width), dtype=torch.int16, device=w.device)
                for factor in factors:
                    trial = (anchor.float() * factor).half().clamp_min(2**-24)
                    expanded = (trial[:, None].float() * relative[codes]).repeat_interleave(
                        groupsize, -1
                    )
                    x = (target / expanded).reshape(-1, 256).contiguous()
                    metric = expanded * h.sqrt()[None, :]
                    metric = (
                        (metric / metric.square().mean(-1, keepdim=True).sqrt())
                        .reshape(-1, 256)
                        .contiguous()
                    )
                    values, states, _ = encode(x, metric, lut, k)
                    reconstruction = (values.reshape_as(target) * expanded).half().float()
                    loss = ((target - reconstruction).square() * h).sum(-1)
                    take = loss < best_loss
                    best_loss[take] = loss[take]
                    best_anchor[take] = trial[take]
                    best_states[take] = states.reshape_as(best_states)[take]
            packed = pack(best_states, k).cpu().numpy().reshape(len(ids), -1)
            anchors = best_anchor.cpu().numpy().view(np.uint16).astype(np.uint64)
            descriptors[host_ids] = np.uint64(k - 1) << np.uint64(32) | anchors << np.uint64(40)
            descriptors[host_ids] |= np.uint64(16 - state_width) << np.uint64(56)
            if codes is not None:
                selectors[host_ids] = codes.byte().cpu().numpy()
            if len(unique_profiles) == 1:
                row_payloads = packed.reshape(-1)
            else:
                for i, row in enumerate(host_ids):
                    row_payloads[row] = packed[i]
        lengths = width * widths.astype(np.uint64) // 32
        descriptors |= np.r_[np.uint64(0), np.cumsum(lengths[:-1], dtype=np.uint64)]
        if column_scaling:
            scales = column_scales.cpu().numpy()
        else:
            selectors = selectors.reshape(-1)
            if len(selectors) & 1:
                selectors = np.pad(selectors, (0, 1))
            scales = selectors[::2] | (selectors[1::2] << 4)
        return TQTensor(
            rows, width, groupsize, column_scaling, descriptors, scales, np.concatenate(row_payloads) if isinstance(row_payloads, list) else row_payloads
        )
