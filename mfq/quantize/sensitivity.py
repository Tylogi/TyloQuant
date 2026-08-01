"""逐 tensor 灵敏度分析。

两层度量（开发文档设计 5）：

1. *简单* —— 权重量化误差（SNR / MSE），不需要激活数据。给逐张量 spec 搜索
   （:mod:`mfq.quantize.search_mat`）提供目标函数。
2. *进阶* —— 校准集下量化层输出 hidden 与全精度输出的距离（function-level）。
   待校准器（:mod:`mfq.calibration`）接入后实现；开发文档 v2 §1.11 指出其有
   ~2.65 dB cross-dim headroom，且对 SwiGLU 的正确目标是乘积误差。
"""

from __future__ import annotations

import numpy as np

from mfq.formats.nint import NintSpec
from mfq.quantize import nint_quant
from mfq.utils.tensor import mse, snr


def weight_snr(weight: np.ndarray, spec: NintSpec, axis: int = 0) -> float:
    """按 ``spec`` 量化 ``weight`` 后的 SNR (dB)。"""

    r = nint_quant.dequantize(nint_quant.quantize(weight, spec, axis=axis))
    return snr(weight, r)


def weight_mse(weight: np.ndarray, spec: NintSpec, axis: int = 0) -> float:
    """按 ``spec`` 量化 ``weight`` 后的 MSE。"""

    r = nint_quant.dequantize(nint_quant.quantize(weight, spec, axis=axis))
    return mse(weight, r)


def output_distance(out_fp: np.ndarray, out_q: np.ndarray) -> float:
    """量化层输出 hidden（``out_q``）与全精度输出（``out_fp``）的距离。

    function-level 度量，待校准器接入后实现。返回归一化 L2 距离。
    """

    raise NotImplementedError
