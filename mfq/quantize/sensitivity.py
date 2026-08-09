"""Per-tensor sensitivity analysis.

Two levels of metrics (development design 5):

1. *Simple* -- weight quantization error (SNR / MSE), requiring no activation data. This provides the objective
   for per-tensor spec search (:mod:`mfq.quantize.search_mat`).
2. *Advanced* -- function-level distance between quantized-layer hidden outputs and full-precision outputs on a calibration set.
   This will be implemented after integrating the calibrator (:mod:`mfq.calibration`). Development documentation v2 section 1.11
   identifies about 2.65 dB of cross-dimension headroom and product error as the correct objective for SwiGLU.
"""

from __future__ import annotations

import numpy as np

from mfq.formats.nint import NintSpec
from mfq.quantize import nint_quant
from mfq.utils.tensor import mse, snr


def weight_snr(weight: np.ndarray, spec: NintSpec, axis: int = 0) -> float:
    """SNR in dB after quantizing ``weight`` according to ``spec``."""

    r = nint_quant.dequantize(nint_quant.quantize(weight, spec, axis=axis))
    return snr(weight, r)


def weight_mse(weight: np.ndarray, spec: NintSpec, axis: int = 0) -> float:
    """MSE after quantizing ``weight`` according to ``spec``."""

    r = nint_quant.dequantize(nint_quant.quantize(weight, spec, axis=axis))
    return mse(weight, r)


def output_distance(out_fp: np.ndarray, out_q: np.ndarray) -> float:
    """Distance between quantized-layer hidden output ``out_q`` and full-precision output ``out_fp``.

    A function-level metric to be implemented after calibrator integration. Returns normalized L2 distance.
    """

    raise NotImplementedError
