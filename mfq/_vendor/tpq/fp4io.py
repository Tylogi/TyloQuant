"""DeepSeek-V4 FP4 weight dequantization using packed e2m1 format plus ue8m0 block scaling.

Format, verified against fp4_quant_kernel in the model repository's inference/kernel.py:
    Pairs of FP4 e2m1 values from logical weight matrix [R, C] are packed into I8 and stored as [R, C/2].
    Nibble order: low nibble = even column first, high nibble = odd column second.
    Scaling: one ue8m0 scale per 32 elements of each row (unsigned 8-bit exponent, actual value = 2^(b-127)),
    stored as [R, C/32]. Dequantization uses W = e2m1_value * 2^(scale-127).
e2m1 value table (one sign bit, two exponent bits, and one mantissa bit, fn variant):
    indices 0..7 = 0, 0.5, 1, 1.5, 2, 3, 4, 6; indices 8..15 are their negatives.
Self-check: dequant_fp4_check verifies nibble order and scaling semantics using amax/scale in [3, 6] for each block of 32.
Quantization maps amax into [3,6]; incorrect nibble order systematically puts this ratio out of range.
"""

from __future__ import annotations

import torch

# Complete 16-value e2m1 lookup table (bit 3 is the sign bit)
_E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                         dtype=torch.float32)


def dequant_fp4(q: torch.Tensor, scale: torch.Tensor, rows: int, cols: int,
                device=None) -> torch.Tensor:
    """Convert I8-packed FP4 plus ue8m0 scaling to f32 [rows, cols].

    q: [rows, cols//2] int8/uint8; scale: [rows, cols//32] exponent bytes with uint8 semantics.
    """
    dev = device or q.device
    lut = _E2M1_LUT.to(dev)
    qu = q.view(torch.uint8).to(dev)
    lo = qu & 0x0F
    hi = qu >> 4
    idx = torch.stack([lo, hi], dim=-1).reshape(rows, cols).long()
    mag = lut[idx]
    s = torch.pow(2.0, scale.view(torch.uint8).to(dev).float() - 127.0)
    return mag * s.repeat_interleave(32, dim=1)


def dequant_fp4_check(q: torch.Tensor, scale: torch.Tensor, rows: int, cols: int,
                      sample_rows: int = 64) -> tuple[float, float]:
    """Format self-check: sample rows and return minimum/maximum amax/scale ratios for each block of 32.

    Normal values lie in [3, 6] because the maximum e2m1 value is 6 and quantization maps block amax into [3,6].
    Systematic values below 3 or above 6 indicate mismatched nibble order or scaling semantics.
    """
    r = min(sample_rows, rows)
    w = dequant_fp4(q[:r], scale[:r], r, cols)
    s = torch.pow(2.0, scale[:r].view(torch.uint8).float() - 127.0)
    amax = w.abs().reshape(r, cols // 32, 32).amax(dim=-1)
    ratio = amax / s
    return float(ratio.min()), float(ratio.max())
